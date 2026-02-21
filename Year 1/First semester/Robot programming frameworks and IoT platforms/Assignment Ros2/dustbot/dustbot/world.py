from dustbot_interface.msg import GridPosition
from dustbot_interface.srv import GarbagePickUp , SetDirection

import rclpy 

from rclpy.node import Node

import random

class WorldNode(Node):
    
    def __init__(self):
        
        super().__init__("world_node")
        self.close_future = rclpy.Future()
        ## Parameters for the world
        self.declare_parameter("N", 10)
        self.declare_parameter("P", 5)
        self.n = self.get_parameter("N").get_parameter_value().integer_value
        self.p = self.get_parameter("P").get_parameter_value().integer_value

        self.get_logger().info(f"World paramaters are N = '{self.n}' and P = '{self.p}'")
        
        # Movement
        self.dir_x = 0
        self.dir_y = 0

        ## Counter PickUp
        self.PickUp = 0
        
        ## Param Robot
        self.robot_x = 0 # Starting position
        self.robot_y = 0 # Same
        
        ## Garbage position
        self.garbage_x = random.randint(0,self.n-1)
        self.garbage_y = random.randint(0,self.n-1)        
        
        # New Position 
        self.response_garbage_clbk = False
        self.timer_period = 1
        
        self.garbage_pos = self.create_publisher(GridPosition,"/dustbot/garbage_position",10)
        self.robot_pos = self.create_publisher(GridPosition,"/dustbot/global_position",10)

        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        ## Service 
        self.srv_pickup = self.create_service(GarbagePickUp,"/dustbot/load_garbage",self.garbage_pickup_clbck)
        
        self.srv_direction = self.create_service(SetDirection,"/dustbot/set_direction",self.set_direction_clbck)
    
    def timer_callback(self):
        # Control if the garbage_service is been called 
        msg_garbage = GridPosition()
        msg_robot = GridPosition()
        
        if self.response_garbage_clbk: # He said that this part has to go directly in the callback

            self.PickUp += 1
            self.get_logger().info(f"The position is reached, the new number of pickup is: '{self.PickUp}'")
            if self.PickUp < self.p:
                self.garbage_x = random.randint(0,self.n-1)
                self.garbage_y = random.randint(0,self.n-1)

                self.response_garbage_clbk = False

            # Finish condition, error he said that im wasting bit, so you have to use an unsigned(?)
            else:
                msg_garbage.x = -1
                msg_garbage.y = -1
                
                self.garbage_pos.publish(msg_garbage)
                self.get_logger().info(f"Number of total pickup reached. Congrats!!!")
          
                self.close_future.set_result(True)
                return     
      
        if 0<=self.robot_x+self.dir_x<self.n and 0<=self.robot_y+self.dir_y<self.n:
            self.robot_x += self.dir_x
            self.robot_y += self.dir_y
        else:
            self.dir_x = 0
            self.dir_y = 0
        
        msg_garbage.x = self.garbage_x
        msg_garbage.y = self.garbage_y

        msg_robot.x = self.robot_x
        msg_robot.y = self.robot_y
         
        self.get_logger().info(f"Robot position: ('{msg_robot.x}','{msg_robot.y}') Garbage position: ('{msg_garbage.x}','{msg_garbage.y}')")
        
        # Publish the Robot position and Garbage 
        self.garbage_pos.publish(msg_garbage)
        self.robot_pos.publish(msg_robot)

    def garbage_pickup_clbck(self,request,response):
        
        # Control if the position is really reached
        
        if self.robot_x == self.garbage_x and self.robot_y == self.garbage_y:
            self.dir_x = 0
            self.dir_y = 0  
            self.response_garbage_clbk = True
            response.success = True
        
        else :
            
            response.success = False
        
        return response
    
    def set_direction_clbck(self,request,response):
        
        # 0 = no movement, x = 1 Est, x = -1 Ovest, y = 1 South, y = -1 North
        if 0<=self.robot_x+request.movement_x<self.n and 0<=self.robot_y+request.movement_y<self.n:
            self.dir_x = request.movement_x
            self.dir_y = request.movement_y

            self.get_logger().info(f"The new direction is:('{self.dir_x}','{self.dir_y}')")

            response.success = True
        else:
            self.dir_x = 0
            self.dir_y = 0
            response.success = False
        return response

def main():
    rclpy.init()

    world_node = WorldNode()

    rclpy.spin_until_future_complete(world_node,world_node.close_future) # The only right way for
    # him is to use a while loop with spin once and a condition to go out, also use rclpy.ok before spin
    # something similar in the slide
    
    world_node.destroy_node()
    rclpy.shutdown()
        
if __name__ == "__main__":
    main()