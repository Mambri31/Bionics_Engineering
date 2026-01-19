// This #include statement was automatically added by the Particle IDE.
#include <Grove-Ultrasonic-Ranger.h>

// Include Particle Device OS APIs
#include "Particle.h"

// Let Device OS manage the connection to the Particle Cloud
SYSTEM_MODE(AUTOMATIC);

// Run the application and system concurrently in separate threads
SYSTEM_THREAD(ENABLED);

// Show system, cloud connectivity, and application logs over USB
// View logs with CLI using 'particle serial monitor --follow'
SerialLogHandler logHandler(LOG_LEVEL_INFO);


//LED blue (D7)
#define LEDD7 D7
  
// button on D2
const int buttonPin = 2;
int buttonState = 0; 

// light sensor on A0
#define LIGHTSENSOR A0

// ultrasound in D4
Ultrasonic ultrasonic(4);

// buzzer on A2
#define BUZZER A2

// angle sensor on A4
#define ROTARY_ANGLE_SENSOR A4
#define ADC_REF 3.3 
#define GROVE_VCC 3.3
#define FULL_ANGLE 300



// setup() runs once, when the device is first turned on
void setup() {
    // Put initialization like pinMode and begin functions here
  
    //serial communication
    Serial.begin(9600);

    //LED blue (D7)
    pinMode(LEDD7, OUTPUT);
    digitalWrite(LEDD7, LOW);
    
    // button
    pinMode(buttonPin, INPUT);
    
    // light sensor
    pinMode(LIGHTSENSOR, INPUT);
    
    // buzzer
    pinMode(BUZZER, OUTPUT);
    
    // angle sensor
    pinMode(ROTARY_ANGLE_SENSOR, INPUT);


}

// loop() runs over and over again, as quickly as it can execute.
void loop() {
  // The core of your code will likely live here.

  // Example: Publish event to cloud every 10 seconds. Uncomment the next 3 lines to try it!
  // Log.info("Sending Hello World to the cloud!");
  // Particle.publish("Hello world!");
  // delay( 10 * 1000 ); // milliseconds and blocking - see docs for more info!
  
    // button and D7 led
    
    buttonState = digitalRead(buttonPin);
 

        if (buttonState == HIGH) {
            digitalWrite(LEDD7, HIGH);
            }
            else {
                digitalWrite(LEDD7, LOW);
                }
                
  
  
    // button, D7 led and serial
    int value = analogRead(LIGHTSENSOR);

    Serial.println("The value of the light sensor is:");
    Serial.println(value);
    delay(1000); 
    
    // ultrasound
    long RangeInInches;
    long RangeInCentimeters;
 
    //Serial.println("The distance to obstacles in front is: ");
    //RangeInInches = ultrasonic.MeasureInInches();
    //Serial.print(RangeInInches);//0~157 inches
    //Serial.println(" inch");
    //delay(250);
 
    RangeInCentimeters = ultrasonic.MeasureInCentimeters();
    Serial.println("The value of the US sensor is:");
    Serial.print(RangeInCentimeters);//0~400cm
    Serial.println(" cm");
    delay(1000);
    
    //buzzer
    digitalWrite(BUZZER, HIGH);
    delay (1000);
    digitalWrite(BUZZER, LOW);
    
    // angle sensor
    float voltage;
    int sensor_value = analogRead(ROTARY_ANGLE_SENSOR);
    voltage = (float)sensor_value*ADC_REF/4095; //12bit resolution
    float degrees = (voltage*FULL_ANGLE)/GROVE_VCC;
    Serial.println("The angle between the mark and the starting position:");
    Serial.println(degrees);
    delay(1000);
    
    
}