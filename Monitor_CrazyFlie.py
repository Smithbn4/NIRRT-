import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger

URI = 'radio://0/80/2M/E7E7E7E7E7'

cflib.crtp.init_drivers()

log_conf = LogConfig(name='FlowHeight', period_in_ms=100)
log_conf.add_variable('stabilizer.roll', 'float')
log_conf.add_variable('stabilizer.pitch', 'float')
log_conf.add_variable('stabilizer.yaw', 'float')
log_conf.add_variable('range.zrange', 'uint16_t')
log_conf.add_variable('stateEstimate.z', 'float')

with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
    # Everything using 'scf' must be indented inside this block
    is_flow_attached = scf.cf.param.get_value('deck.bcFlow2')
    print(f"Flow deck v2 detected: {is_flow_attached}")

    with SyncLogger(scf, log_conf) as logger:
        for log_entry in logger:
            timestamp = log_entry[0]
            data = log_entry[1]
            
            raw_mm = data['range.zrange']
            est_m = data['stateEstimate.z']
            
            print(f"[{timestamp}] "
                  f"Raw Range: {raw_mm} mm | "
                  f"Est Height: {est_m:.3f} m | "
                  f"Roll: {data['stabilizer.roll']:.1f}, "
                  f"Pitch: {data['stabilizer.pitch']:.1f}, "
                  f"Yaw: {data['stabilizer.yaw']:.1f}")
