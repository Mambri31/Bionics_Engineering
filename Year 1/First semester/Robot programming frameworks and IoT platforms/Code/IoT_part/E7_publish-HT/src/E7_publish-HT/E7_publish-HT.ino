// This #include statement was automatically added by the Particle IDE.
#include <Adafruit_DHT_Particle.h>

// Include Particle Device OS APIs
#include "Particle.h"

// Let Device OS manage the connection to the Particle Cloud
SYSTEM_MODE(AUTOMATIC);

// Run the application and system concurrently in separate threads
SYSTEM_THREAD(ENABLED);

// Show system, cloud connectivity, and application logs over USB
// View logs with CLI using 'particle serial monitor --follow'
SerialLogHandler logHandler(LOG_LEVEL_INFO);


#define DHTPIN 	D2
#define DHTTYPE DHT11		

DHT dht(DHTPIN, DHTTYPE);

float currentTemp = 0;
float currentHum = 0;


void setup() {

dht.begin();

}

void loop() {
    
    currentTemp = (float)dht.getTempCelcius();
    currentHum = (float)dht.getHumidity();
    
    Particle.publish("Temperature", String::format("%.1f", currentTemp), PRIVATE);
    delay (1000);
    Particle.publish("Humidity", String::format("%.1f", currentHum), PRIVATE);
    delay (1000);

}
    