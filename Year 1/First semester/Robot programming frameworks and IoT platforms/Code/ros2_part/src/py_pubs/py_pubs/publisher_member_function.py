import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__("minimal_publisher")
        self.publisher_ =self.create_publisher(msg_type=String, topic="topic", qos_profile=10)
        timer_period = 0.5
        self.time=self.create_timer(timer_period_sec=timer_period, callback=self.timer_callback)
        self.counter = 0

    def timer_callback(self):
        msg = String()
        msg.data= f"Hello World: {self.counter}"

        self.publisher_.publish(msg)

        self.get_logger().info(f"Published: {msg.data}")
        self.counter += 1

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()

#common structure fon any ROS node