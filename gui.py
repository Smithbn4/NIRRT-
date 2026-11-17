
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import plan as planning
from path_planning_classes.rrt_base_2d import RRTBase2D
from path_planning_utils.Astar_with_clearance import AStar
from path_planning_utils.custom_map import MapSpec, Obstacle, integer
from path_planning_utils.heatmap import probability_rgb


class MapCanvas(tk.Canvas):
    """Aspect-preserving occupancy view, using the actual solver mask."""
    def __init__(self, parent):
        super().__init__(parent, background='#e9eef3', highlightthickness=0,
                         width=330, height=360)
        self.grid_map = self.spec = None
        self.paths = []
        self.heatmap = None
        self.photo = None
        self.bind('<Configure>', lambda event: self.redraw())

    def show(self, grid, spec, paths=(), heatmap=None):
        self.grid_map, self.spec, self.paths = grid, spec, paths
        if heatmap is not None and heatmap.shape != grid.binary_mask.shape:
            raise ValueError('Heatmap dimensions must match the occupancy map')
        self.heatmap = heatmap
        self.redraw()

    def redraw(self):
        self.delete('all')
        if self.grid_map is None:
            self.create_text(20, 30, anchor='nw', text='Create a map to preview it.')
            return
        width, height = self.spec.width, self.spec.height
        scale = min(max(1, self.winfo_width()-56)/width,
                    max(1, self.winfo_height()-56)/height)
        pw, ph = max(1, int(width*scale)), max(1, int(height*scale))
        ox, oy = (self.winfo_width()-pw)/2, (self.winfo_height()-ph)/2
        # Nearest-neighbor display only; planning always uses full resolution.
        xs = np.minimum((np.arange(pw)*width/pw).astype(int), width-1)
        ys = np.minimum((np.arange(ph)*height/ph).astype(int), height-1)
        free = self.grid_map.binary_mask[np.ix_(ys, xs)] != 0
        valid = self.grid_map.valid_mask[np.ix_(ys, xs)]
        rgb = np.full((ph, pw, 3), 255, dtype=np.uint8)
        if self.heatmap is not None:
            rgb = probability_rgb(self.heatmap[np.ix_(ys, xs)])
        rgb[~valid] = (218, 224, 231)
        rgb[~free] = (40, 48, 62)
        ppm = 'P6\n{} {}\n255\n'.format(pw, ph).encode()+rgb.tobytes()
        self.photo = tk.PhotoImage(data=ppm, format='PPM')
        self.create_image(ox, oy, image=self.photo, anchor='nw')
        self.create_rectangle(ox, oy, ox+pw, oy+ph, outline='#8796a6')

        def position(point):
            return ox+(point[0]+0.5)*pw/width, oy+(point[1]+0.5)*ph/height

        for path, color, dash in self.paths:
            if len(path) > 1:
                coordinates = [v for point in path for v in position(point)]
                self.create_line(*coordinates, fill=color, width=2.5, dash=dash)
        for point, color, label in [(self.spec.start, '#118458', 'S'),
                                    (self.spec.goal, '#a85f00', 'G')]:
            x, y = position(point)
            self.create_oval(x-5, y-5, x+5, y+5, fill=color, outline='white')
            self.create_text(x+8, y-8, text=label, fill=color, anchor='sw')
        self.create_text(ox, oy-12, text='(0, 0)', anchor='w', fill='#506070')
        self.create_text(ox+pw, oy+ph+13, text='({}, {})'.format(width-1, height-1),
                         anchor='e', fill='#506070')


class PlanningApp:
    def __init__(self, root):
        self.root = root
        root.title('Custom map planner — Neural NRRT* and A*')
        root.geometry('1320x960')
        root.minsize(1080, 840)
        self.events = queue.Queue()
        self.cancel_event = threading.Event()
        self.busy = False
        self.grid = self.spec = None
        self.obstacles = []
        self.results = {}
        self.history = []
        self.heatmap = None
        self.heatmap_seconds = None
        self.neural_error = None
        self._network = None  # Loaded once, reused by subsequent worker runs.
        self.inputs = {}
        self.editable = []
        self.status = tk.StringVar(value='Set map dimensions, endpoints, and up to 10 obstacles.')
        self._build_ui()
        for name in ('width', 'height', 'sx', 'sy', 'gx', 'gy', 'clearance'):
            self.inputs[name].trace_add('write', lambda *args: self._invalidate())
        self.preview()
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(80, self._poll)

    def _entry(self, parent, label, name, default, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=5, pady=3)
        variable = tk.StringVar(value=str(default))
        entry = ttk.Entry(parent, textvariable=variable, width=16)
        entry.grid(row=row, column=1, sticky='ew', padx=5, pady=3)
        self.inputs[name] = variable
        self.editable.append(entry)
        return entry

    def _button(self, parent, text, command):
        button = ttk.Button(parent, text=text, command=command)
        self.editable.append(button)
        return button

    def _build_ui(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        # Scrollable form keeps every control reachable on smaller displays.
        sidebar = ttk.Frame(self.root, padding=8)
        sidebar.grid(row=0, column=0, sticky='ns')
        sidebar.rowconfigure(0, weight=1)
        scroller = tk.Canvas(sidebar, width=325, highlightthickness=0)
        scroller.grid(row=0, column=0, sticky='ns')
        scroll = ttk.Scrollbar(sidebar, command=scroller.yview)
        scroll.grid(row=0, column=1, sticky='ns')
        scroller.configure(yscrollcommand=scroll.set)
        controls = ttk.Frame(scroller)
        window = scroller.create_window(0, 0, window=controls, anchor='nw', width=325)
        controls.bind('<Configure>', lambda event: scroller.configure(scrollregion=scroller.bbox('all')))
        scroller.bind('<Configure>', lambda event: scroller.itemconfigure(window, width=event.width))

        map_box = ttk.LabelFrame(controls, text='Map and endpoints (pixels)', padding=5)
        map_box.pack(fill='x', pady=(0, 8))
        fields = [('Map width', 'width', 224), ('Map height', 'height', 224),
                  ('Start x', 'sx', 15), ('Start y', 'sy', 15),
                  ('Goal x', 'gx', 208), ('Goal y', 'gy', 208),
                  ('Clearance radius', 'clearance', 0)]
        for row, (label, name, value) in enumerate(fields):
            self._entry(map_box, label, name, value, row)
        ttk.Label(map_box, text='Origin: top-left. x → right, y → down.\nInteger coordinates; width/height ≥ 2.',
                  justify='left').grid(row=len(fields), column=0, columnspan=2, sticky='w', padx=5, pady=5)

        obstacle_box = ttk.LabelFrame(controls, text='Obstacles (maximum 10)', padding=5)
        obstacle_box.pack(fill='x', pady=(0, 8))
        self.kind = tk.StringVar(value='Rectangle')
        self.kind_box = ttk.Combobox(obstacle_box, textvariable=self.kind,
                                    values=['Rectangle', 'Circle'], state='readonly', width=13)
        ttk.Label(obstacle_box, text='Shape').grid(row=0, column=0, sticky='w', padx=5)
        self.kind_box.grid(row=0, column=1, sticky='ew', padx=5)
        self.kind_box.bind('<<ComboboxSelected>>', self._shape_changed)
        self._entry(obstacle_box, 'x (left / center)', 'ox', 70, 1)
        self._entry(obstacle_box, 'y (top / center)', 'oy', 70, 2)
        self.size_label = ttk.Label(obstacle_box, text='Length along x')
        self.size_entry = self._entry(obstacle_box, '', 'size', 55, 3)
        self.size_label.grid(row=3, column=0, sticky='w', padx=5)
        self.obstacle_width_entry = self._entry(obstacle_box, 'Width along y', 'ow', 25, 4)
        self._button(obstacle_box, 'Add obstacle', self.add_obstacle).grid(row=5, column=0, columnspan=2, sticky='ew', pady=5)
        self.obstacle_list = tk.Listbox(obstacle_box, height=5, width=34, exportselection=False)
        self.obstacle_list.grid(row=6, column=0, columnspan=2, sticky='ew')
        self._button(obstacle_box, 'Remove selected', self.remove_obstacle).grid(row=7, column=0, pady=5)
        self._button(obstacle_box, 'Clear all', self.clear_obstacles).grid(row=7, column=1, pady=5)
        self.obstacle_count = ttk.Label(obstacle_box, text='0 / 10 obstacles')
        self.obstacle_count.grid(row=8, column=0, columnspan=2, sticky='w')

        solver_box = ttk.LabelFrame(controls, text='NRRT* search settings', padding=5)
        solver_box.pack(fill='x', pady=(0, 8))
        for row, (label, name, value) in enumerate([
                ('Iteration budget', 'iterations', 1500), ('Step length', 'step', 15),
                ('Random seed', 'seed', 100), ('Neural sampling rate', 'neural_rate', 0.5)]):
            self._entry(solver_box, label, name, value, row)
        self.stop_first = tk.BooleanVar(value=False)
        check = ttk.Checkbutton(solver_box, text='Stop at first NRRT* path', variable=self.stop_first)
        check.grid(row=4, column=0, columnspan=2, sticky='w', pady=5)
        self.editable.append(check)
        ttk.Label(solver_box, text='0.5 = 50% neural corridor samples\nand 50% global free-space samples.',
                  justify='left').grid(row=5, column=0, columnspan=2, sticky='w', padx=5, pady=4)
        buttons = ttk.Frame(controls)
        buttons.pack(fill='x')
        self._button(buttons, 'Preview map', self.preview).pack(side='left', expand=True, fill='x')
        self._button(buttons, 'Solve both', self.solve).pack(side='left', expand=True, fill='x', padx=3)
        self.cancel_button = ttk.Button(buttons, text='Cancel', command=self.cancel, state='disabled')
        self.cancel_button.pack(side='left')

        main = ttk.Frame(self.root, padding=(4, 8, 12, 8))
        main.grid(row=0, column=1, sticky='nsew')
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)
        maps = ttk.Frame(main)
        maps.grid(row=0, column=0, sticky='nsew')
        maps.columnconfigure((0, 1), weight=1, uniform='map')
        maps.rowconfigure(0, weight=1)
        self.views = []
        for column, title in enumerate(['Custom map (unsolved)', 'Solved map — NRRT* red / A* blue dashed']):
            box = ttk.LabelFrame(maps, text=title, padding=4)
            box.grid(row=0, column=column, sticky='nsew', padx=(0, 6))
            view = MapCanvas(box)
            view.pack(fill='both', expand=True)
            self.views.append(view)
        ttk.Label(main, text='S: start   G: goal   Dark: obstacle   Gray: clearance margin',
                  foreground='#506070').grid(row=1, column=0, sticky='w', pady=5)
        heatmap_box = ttk.Frame(main)
        heatmap_box.grid(row=2, column=0, sticky='ew', pady=(0, 5))
        self.show_heatmap = tk.BooleanVar(value=True)
        ttk.Checkbutton(heatmap_box, text='Show neural probability heatmap',
                        variable=self.show_heatmap, command=self._show_solution).pack(side='left')
        legend = tk.Canvas(heatmap_box, width=200, height=32, highlightthickness=0)
        legend.pack(side='left', padx=10)
        for index, color in enumerate(probability_rgb(np.linspace(0, 1, 160))):
            legend.create_line(20+index, 2, 20+index, 14,
                               fill='#{:02x}{:02x}{:02x}'.format(*color))
        for x, label in [(20, '0'), (100, '0.5'), (180, '1')]:
            legend.create_text(x, 24, text=label)
        self.heatmap_status = tk.StringVar(value='Neural path probability; appears after Solve both.')
        ttk.Label(main, textvariable=self.heatmap_status, wraplength=740,
                  foreground='#506070').grid(row=3, column=0, sticky='w', pady=(0, 5))
        self.metrics = ttk.Treeview(main, columns=('metric', 'nrrt', 'astar'), show='headings', height=10,
                                   selectmode='none')
        for name, title, width in [('metric', 'Measured metric', 220), ('nrrt', 'NRRT* (neural-guided)', 235), ('astar', 'A* (8 neighbors)', 200)]:
            self.metrics.heading(name, text=title)
            self.metrics.column(name, width=width, minwidth=100)
        self.metrics.grid(row=4, column=0, sticky='ew')
        self._update_metrics()
        chart_box = ttk.LabelFrame(main, text='NRRT* convergence — best path length over search time', padding=3)
        chart_box.grid(row=5, column=0, sticky='ew', pady=8)
        self.chart = tk.Canvas(chart_box, height=130, background='white', highlightthickness=0)
        self.chart.pack(fill='x')
        self.chart.bind('<Configure>', lambda event: self._draw_history())
        ttk.Label(main, text='Last improvement is the time the final best length was first reached; it does not prove convergence.\n'
                            'Search times exclude model setup; inclusive times add it. Lengths are in pixels.',
                  wraplength=740, foreground='#506070').grid(row=6, column=0, sticky='w')
        ttk.Label(self.root, textvariable=self.status, padding=8, wraplength=1050).grid(row=1, column=0, columnspan=2, sticky='ew')

    def _shape_changed(self, event=None):
        circle = self.kind.get() == 'Circle'
        self.size_label.configure(text='Radius' if circle else 'Length along x')
        self.obstacle_width_entry.configure(state='disabled' if circle or self.busy else 'normal')

    def _number(self, name, label, minimum=0):
        return integer(self.inputs[name].get(), label, minimum)

    def _read_spec(self):
        spec = MapSpec(self._number('width', 'Map width', 2), self._number('height', 'Map height', 2),
                       (self._number('sx', 'Start x'), self._number('sy', 'Start y')),
                       (self._number('gx', 'Goal x'), self._number('gy', 'Goal y')),
                       tuple(self.obstacles), self._number('clearance', 'Clearance'))
        spec.validate()
        return spec

    def _error(self, error):
        messagebox.showerror('Check input', str(error) or type(error).__name__, parent=self.root)

    def _refresh_obstacles(self):
        self.obstacle_list.delete(0, 'end')
        for obstacle in self.obstacles:
            dimensions = 'r={}'.format(obstacle.size) if obstacle.kind == 'Circle' else '{} × {}'.format(obstacle.size, obstacle.width)
            self.obstacle_list.insert('end', '{} ({}, {}) {}'.format(obstacle.kind, obstacle.x, obstacle.y, dimensions))
        self.obstacle_count.configure(text='{} / 10 obstacles'.format(len(self.obstacles)))
        self._invalidate()

    def add_obstacle(self):
        try:
            if len(self.obstacles) >= 10:
                raise ValueError('Remove an obstacle before adding another (maximum 10).')
            obstacle = Obstacle(self.kind.get(), self._number('ox', 'Obstacle x'), self._number('oy', 'Obstacle y'),
                                self._number('size', 'Length / radius', 1),
                                self._number('ow', 'Rectangle width', 1) if self.kind.get() == 'Rectangle' else 1)
            obstacle.validate(self._number('width', 'Map width', 2), self._number('height', 'Map height', 2))
            self.obstacles.append(obstacle)
            self._refresh_obstacles()
        except ValueError as error:
            self._error(error)

    def remove_obstacle(self):
        selected = self.obstacle_list.curselection()
        if selected:
            del self.obstacles[selected[0]]
            self._refresh_obstacles()

    def clear_obstacles(self):
        self.obstacles.clear()
        self._refresh_obstacles()

    def _invalidate(self):
        self.grid = self.spec = None
        self.results.clear()
        self.history.clear()
        self.heatmap = None
        self.heatmap_seconds = None
        self.neural_error = None
        self.heatmap_status.set('Neural path probability; appears after Solve both.')
        for view in self.views:
            view.show(None, None)
        self._update_metrics()
        self._draw_history()
        self.status.set('Map edited. Preview or solve to apply the current inputs.')

    def preview(self):
        if self.busy:
            return
        try:
            spec = self._read_spec()
            grid = spec.build_grid()
            self._invalidate()
            self.spec, self.grid = spec, grid
            for view in self.views:
                view.show(grid, spec)
            self.status.set('Preview: {} × {} pixels, {} obstacles. Ready to solve.'.format(spec.width, spec.height, len(spec.obstacles)))
        except (ValueError, MemoryError) as error:
            self._error(error)

    def _set_busy(self, busy):
        self.busy = busy
        for widget in self.editable:
            widget.configure(state='disabled' if busy else 'normal')
        self.kind_box.configure(state='disabled' if busy else 'readonly')
        self.obstacle_list.configure(state='disabled' if busy else 'normal')
        self.cancel_button.configure(state='normal' if busy else 'disabled')
        self._shape_changed()

    def solve(self):
        if self.busy:
            return
        try:
            spec = self._read_spec()
            iterations = self._number('iterations', 'Iteration budget', 1)
            seed = self._number('seed', 'Random seed')
            if seed > 2**32-1:
                raise ValueError('Random seed must be between 0 and 4294967295')
            step = float(self.inputs['step'].get())
            if not np.isfinite(step) or step <= 0:
                raise ValueError('Step length must be finite and positive')
            neural_rate = float(self.inputs['neural_rate'].get())
            if not 0 < neural_rate <= 1:
                raise ValueError('NRRT* neural sampling rate must be greater than 0 and at most 1')
        except ValueError as error:
            self._error(error)
            return
        self._invalidate()
        self.cancel_event.clear()
        self._set_busy(True)
        self.status.set('Building the full-resolution occupancy map…')
        threading.Thread(target=self._worker,
                         args=(spec, iterations, step, seed, self.stop_first.get(), neural_rate), daemon=True).start()

    def _worker(self, spec, iterations, step, seed, stop_first, neural_rate):
        """Queue plain data; every Tk operation stays on the main thread."""
        try:
            grid = spec.build_grid()
            self.events.put(('map', (spec, grid)))
            if self.cancel_event.is_set():
                return
            self.events.put(('status', 'Solving A*…'))
            # Both algorithms use exactly the same inflated occupancy mask.
            astar = AStar(spec.start, spec.goal, grid.valid_mask, 0, 'euclidean')
            path, _ = astar.searching(cancel_check=self.cancel_event.is_set,
                                     progress_callback=lambda count: self.events.put(('status', 'Solving A*: {:,} cells visited…'.format(count))))
            path = np.asarray(path[::-1], dtype=float).reshape(-1, 2)
            self._validate_path(grid, spec, path)
            self.events.put(('result', ('astar', dict(path=path, seconds=astar.planning_seconds,
                             length=RRTBase2D.get_path_len(path), cancelled=astar.cancelled,
                             visited=len(astar.CLOSED)))))
            if self.cancel_event.is_set():
                return
            self.events.put(('status', 'Preparing NRRT*: loading model and predicting its sampling corridor…'))
            # Both entry points use plan.py's pipeline. Constructor inference
            # supplies both the planner's sampling corridor and the overlay.
            setup = planning.prepare_planner(grid, spec.start, spec.goal, step_len=step,
                                              iterations=iterations, seed=seed, device='cpu',
                                              neural_sample_rate=neural_rate, wrapper=self._network)
            self._network = setup.wrapper
            planner = setup.planner
            self.events.put(('guidance', dict(scores=planner.path_score, seconds=setup.seconds,
                             points=len(planner.guidance), rate=planner.neural_sample_rate)))
            if self.cancel_event.is_set():
                return
            self.events.put(('status', 'Solving NRRT* with neural-guided sampling…'))

            def progress(current):
                # Copy mutable history before handing it to the UI thread.
                self.events.put(('progress', (current.iterations_run, iterations, list(current.convergence_history))))

            path = planning.run_planner(planner, stop_on_first=stop_first, progress_callback=progress,
                                        cancel_check=self.cancel_event.is_set)
            self.events.put(('result', ('nrrt', dict(path=path, seconds=planner.planning_seconds,
                             length=planner.get_path_len(path), first=planner.first_solution_seconds,
                             best=planner.best_solution_seconds, cancelled=planner.cancelled,
                             iterations=planner.iterations_run, history=list(planner.convergence_history),
                             setup_seconds=setup.seconds, total_seconds=setup.seconds+planner.planning_seconds,
                             first_with_setup=(setup.seconds+planner.first_solution_seconds
                                               if planner.first_solution_seconds is not None else None),
                             guidance_points=len(planner.guidance), neural_rate=planner.neural_sample_rate,
                             neural_samples=planner.neural_samples, uniform_samples=planner.uniform_samples))))
        except Exception as error:
            self.events.put(('error', '{}: {}'.format(type(error).__name__, error)))
        finally:
            self.events.put(('done', None))

    @staticmethod
    def _validate_path(grid, spec, path):
        if len(path) and (not np.array_equal(path[0], spec.start) or
                          not np.array_equal(path[-1], spec.goal) or
                          any(grid.is_collision(a, b) for a, b in zip(path[:-1], path[1:]))):
            raise RuntimeError('Planner returned an invalid path')

    def _poll(self):
        try:
            while True:
                event, data = self.events.get_nowait()
                if event == 'map':
                    self.spec, self.grid = data
                    for view in self.views:
                        view.show(self.grid, self.spec)
                elif event == 'status':
                    self.status.set(data)
                elif event == 'guidance':
                    self.heatmap, self.heatmap_seconds = data['scores'], data['seconds']
                    if data['points']:
                        self.heatmap_status.set('NRRT* guidance: {:,} valid corridor pixels; {:.0%} neural / {:.0%} global sampling. Model + guidance: {:.3f} s.'.format(
                            data['points'], data['rate'], 1-data['rate'], self.heatmap_seconds))
                    else:
                        self.heatmap_status.set('No predicted corridor pixels survive clearance. NRRT* is using its global-sampling fallback. Model + guidance: {:.3f} s.'.format(self.heatmap_seconds))
                    self._show_solution()
                elif event == 'progress':
                    count, total, self.history = data
                    self.status.set('NRRT*: {:,} / {:,} iterations; {} path improvements recorded.'.format(count, total, len(self.history)))
                    self._draw_history()
                elif event == 'result':
                    name, result = data
                    self.results[name] = result
                    if name == 'nrrt':
                        self.history = result['history']
                    self._update_metrics()
                    self._show_solution()
                    self._draw_history()
                elif event == 'error':
                    self.neural_error = data
                    self.heatmap_status.set('NRRT* failed: '+data)
                    self._update_metrics()
                    self.status.set('Search failed: '+data)
                    self._error(data)
                elif event == 'done':
                    self._set_busy(False)
                    if self.cancel_event.is_set():
                        self.status.set('Cancelled. Completed results and any NRRT* path found so far are shown.')
                    elif len(self.results) == 2:
                        self.status.set('Finished: NRRT* (plan.py neural-guided pipeline) compared with A*.')
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    def _show_solution(self):
        paths = [(self.results[key]['path'], color, dash) for key, color, dash in
                 [('nrrt', '#d03b3b', ()), ('astar', '#1768cc', (5, 3))] if key in self.results]
        self.views[1].show(self.grid, self.spec, paths,
                           self.heatmap if self.show_heatmap.get() else None)

    def _update_metrics(self):
        for row in self.metrics.get_children():
            self.metrics.delete(row)
        nrrt, astar = self.results.get('nrrt'), self.results.get('astar')

        def outcome(result, name):
            if result is None:
                if name == 'nrrt' and self.neural_error:
                    return 'Failed (see status)'
                return '—'
            if result['cancelled']:
                return 'Cancelled (path retained)' if len(result['path']) else 'Cancelled'
            if len(result['path']):
                return 'Path found'
            return 'Budget exhausted; no path found' if name == 'nrrt' else 'No grid path exists'

        def seconds(result, key):
            value = result.get(key) if result else None
            return '—' if value is None else '{:.6f} s'.format(value)

        def length(result):
            return '{:.3f} px'.format(result['length']) if result and len(result['path']) else '—'

        astar_solution = seconds(astar, 'seconds') if astar and len(astar['path']) else '—'
        rows = [('Status', outcome(nrrt, 'nrrt'), outcome(astar, 'astar')),
                ('Final path length', length(nrrt), length(astar)),
                ('First solution (search)', seconds(nrrt, 'first'), astar_solution),
                ('Last improvement (search)', seconds(nrrt, 'best'), '—'),
                ('Total search time', seconds(nrrt, 'seconds'), seconds(astar, 'seconds')),
                ('Model + guidance setup', seconds(nrrt, 'setup_seconds'), '—'),
                ('First solution incl. setup', seconds(nrrt, 'first_with_setup'), astar_solution),
                ('Total time incl. setup', seconds(nrrt, 'total_seconds'), seconds(astar, 'seconds')),
                ('Iterations / visited cells', str(nrrt['iterations']) if nrrt else '—', str(astar['visited']) if astar else '—'),
                ('Neural / global sample draws', '{} / {}'.format(nrrt['neural_samples'], nrrt['uniform_samples']) if nrrt else '—', '—')]
        for row in rows:
            self.metrics.insert('', 'end', values=row)

    def _draw_history(self):
        self.chart.delete('all')
        if not self.history:
            self.chart.create_text(20, 25, anchor='nw', text='NRRT* improvements will appear here after a path is found.', fill='#506070')
            return
        width, height = max(200, self.chart.winfo_width()), self.chart.winfo_height()
        left, top, right, bottom = 80, 15, width-20, height-33
        end = self.results.get('nrrt', {}).get('seconds', self.history[-1][1])
        end = max(end, 1e-6)
        lengths = [row[2] for row in self.history]
        low, high = min(lengths), max(lengths)
        margin = max((high-low)*0.12, high*0.01, 0.1)
        low, high = low-margin, high+margin

        def pos(seconds, length):
            return left+(right-left)*seconds/end, bottom-(bottom-top)*(length-low)/(high-low)

        self.chart.create_line(left, top, left, bottom, right, bottom, fill='#8796a6')
        points = []
        previous = self.history[0][2]
        for _, elapsed, length in self.history:
            points.extend(pos(elapsed, previous))
            points.extend(pos(elapsed, length))
            previous = length
        points.extend(pos(end, previous))
        self.chart.create_line(*points, fill='#d03b3b', width=2)
        x, y = pos(self.history[0][1], self.history[0][2])
        self.chart.create_oval(x-3, y-3, x+3, y+3, fill='#d03b3b', outline='')
        for value in [min(lengths), max(lengths)]:
            _, y = pos(0, value)
            self.chart.create_text(left-7, y, text='{:.2f} px'.format(value), anchor='e')
        self.chart.create_text(left, bottom+14, text='0 s', anchor='w')
        self.chart.create_text(right, bottom+14, text='{:.4f} s'.format(end), anchor='e')

    def cancel(self):
        self.cancel_event.set()
        self.cancel_button.configure(state='disabled')
        self.status.set('Cancelling after the current planner operation…')

    def close(self):
        self.cancel_event.set()
        self.root.destroy()


def main():
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise SystemExit('Cannot open Tkinter: {}. Run in a desktop session with Tk installed.'.format(error))
    PlanningApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
