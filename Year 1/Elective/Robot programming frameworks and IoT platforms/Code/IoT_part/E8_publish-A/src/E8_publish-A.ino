// Include Particle Device OS APIs
#include "Particle.h"

// Let Device OS manage the connection to the Particle Cloud
SYSTEM_MODE(AUTOMATIC);

// Run the application and system concurrently in separate threads
SYSTEM_THREAD(ENABLED);

// Show system, cloud connectivity, and application logs over USB
// View logs with CLI using 'particle serial monitor --follow'
SerialLogHandler logHandler(LOG_LEVEL_INFO);

#define potenz_pin A0 //This is where the Angle Sensor is plugged in
#define ADC_ref 3.3 // This is the reference voltage at 3.3V
#define FULL_ANGLE 300 //This is the full range of the Angle Sensor
#define GROVE_VCC 3.3 //This is the VCC of the grove interface


bool flag = false;

int activate(String msg){
    
    if (msg == "ON") flag = true;
    else if (msg == "OFF") flag = false;
    else  {
        digitalWrite(7, HIGH);
        delay (500);
        digitalWrite(7, LOW);
        delay(500);
        digitalWrite(7, HIGH);
        delay (500);
        digitalWrite(7, LOW);
    }
    
    return 0;
}

void setup() {
    
    pinMode(potenz_pin, INPUT); //Set up the Angle sensor pin as input
    pinMode(7, OUTPUT); //Set up the on board led as output
    
    Particle.function("activatate", activate);
    
}

void loop() {
    
    float value, angle;
    
    if (flag == true){
        
        value = (float)analogRead(potenz_pin); //read the voltage from the angle sensors 
        value = value * ADC_ref /4095;
        angle = value * FULL_ANGLE/GROVE_VCC; //converts the analog reading of the voltage in degrees
        
        if (angle < 100)
            Particle.publish ("ANGLE", "LOW", PRIVATE); 

            
        else if (angle < 200)
            Particle.publish ("ANGLE", "MEDIUM", PRIVATE);
            
        else
            Particle.publish ("ANGLE", "HIGH", PRIVATE);
        
        
    }
    delay(1000);
    
}