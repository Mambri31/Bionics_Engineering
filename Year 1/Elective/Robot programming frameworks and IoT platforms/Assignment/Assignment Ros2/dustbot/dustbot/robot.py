from dustbot_interface.msg import GridPosition 
from dustbot_interface.srv import GarbagePickUp , SetDirection

import rclpy 
from rclpy.node import Node



class RobotNode(Node):
    
    def __init__(self):
        
        super().__init__("robot_node")
        self.close_future = rclpy.Future()
        
        self.robot_x = None
        self.robot_y = None
        
        self.garbage_x = None
        self.garbage_y = None
        
        self.garbage_pos = self.create_subscription(GridPosition,"/dustbot/garbage_position",self.garbage_listener_clbk,10)
        self.robot_pos = self.create_subscription(GridPosition,"/dustbot/global_position",self.robot_position_clbk,10)
        
        self.dir_cli = self.create_client(SetDirection,"/dustbot/set_direction")
        self.pickup_cli = self.create_client(GarbagePickUp,"/dustbot/load_garbage")
        
        while not self.dir_cli.wait_for_service(timeout_sec=1):
            self.get_logger().info("Waiting for the /dustbot/set_direction service")
        while not self.pickup_cli.wait_for_service(timeout_sec=1):
            self.get_logger().info("Waiting for /dustbot/load_garbage sevice")

       
        

    def garbage_listener_clbk(self,msg):
        self.garbage_x = msg.x
        self.garbage_y = msg.y
        if self.garbage_x == -1 and self.garbage_y == -1:
            self.get_logger().info(f"I have finished my task!!!!!") 
            self.close_future.set_result(True)
            return 

    def robot_position_clbk(self,msg):
        
        self.robot_x = msg.x
        self.robot_y = msg.y   
        
        if self.garbage_x is None or self.garbage_y is None:
            return
        
        if self.robot_x == self.garbage_x and self.robot_y == self.garbage_y:
            self.garbage_service_clbk()
        else:
            self.dir_service_clbk()
    
    def garbage_service_clbk(self):
        self.req_pickup = GarbagePickUp.Request()
        self.req_pickup.pick = True
        self.future_gb = self.pickup_cli.call_async(self.req_pickup)
        self.future_gb.add_done_callback(self.garbage_pickup_done_clbk) 
    
    def garbage_pickup_done_clbk(self,future_gb): 
        response = future_gb.result()
        if response.success:
            self.get_logger().info(f"Pickup done")
        else: 
            self.get_logger().info(f"The position is wrong impossible to pickup")
    
    def dir_service_clbk(self):
        self.req_mov = SetDirection.Request()
        # OVEST - EST
        if self.robot_x!=self.garbage_x:
            self.req_mov.movement_y = 0
            if self.robot_x>self.garbage_x:
                self.req_mov.movement_x = -1
            elif self.robot_x<self.garbage_x:
                self.req_mov.movement_x = 1
            else:
                self.req_mov.movement_x = 0
        # NORTH-SOUTH
        else:
            self.req_mov.movement_x = 0
            if self.robot_y>self.garbage_y:
                self.req_mov.movement_y = -1
            elif self.robot_y<self.garbage_y:
                self.req_mov.movement_y = 1
            else:
                self.req_mov.movement_y = 0
        
        self.future_mov = self.dir_cli.call_async(self.req_mov)
        self.future_mov.add_done_callback(self._set_direction_done_clbk) 

    def _set_direction_done_clbk(self,future_dir): 
        response = future_dir.result()
        if response.success:
            pass
        else:
            self.get_logger().info(f"Movement out bound")

def main():
    
    rclpy.init()

    robot_node = RobotNode()
    
    rclpy.spin_until_future_complete(robot_node,robot_node.close_future) 
    
    robot_node.destroy_node()
    
    rclpy.shutdown()  
          
if __name__=="__main__":
    main()