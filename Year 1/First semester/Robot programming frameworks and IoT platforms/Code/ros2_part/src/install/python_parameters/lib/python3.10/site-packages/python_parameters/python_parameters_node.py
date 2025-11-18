import rclpy
import rclpy.node
from rcl_interfaces.msg import ParameterDescriptor
 

class MinimalParam(rclpy.node.Node):
    def __init__(self):
        super().__init__('minimal_param_node')

        # declaration is required unless the node is initialized with
     # allow_undeclared_parameters=True
        my_parameter_descriptor = ParameterDescriptor(description='This parameter is mine!')
        self.declare_parameter(name='my_parameter', value='world', descriptor=my_parameter_descriptor)
        self.timer = self.create_timer(1, self.timer_callback)
    def timer_callback(self):
        my_param = self.get_parameter('my_parameter').get_parameter_value().string_value
        self.get_logger().info(f'Hello {my_param}!')
        
        my_new_param = rclpy.parameter.Parameter(
            name='my_parameter',
            type_=rclpy.Parameter.Type.STRING, # optional inferred from value
            value='world'
        )
        all_new_parameters = [my_new_param]
        self.set_parameters(parameter_list=all_new_parameters)


def main():
    rclpy.init()
    node = MinimalParam()
    rclpy.spin(node)
if __name__ == '__main__':
    main()