from example_interfaces.srv import AddTwoInts #import the service definition

import rclpy
from rclpy.node import Node

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_clbck)
    def add_two_ints_clbck(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Incoming request\na: {request.a} b: {request.b}')
        return response
                    
    

def main():
    rclpy.init()
    
    minimal_service = MinimalService() #creation of an instance for the MinimalService class
    try:
        rclpy.spin(minimal_service)
    except KeyboardInterrupt:
        print("Failed")
    finally:
        rclpy.shutdown()

if __name__=="__main__":
    main()

