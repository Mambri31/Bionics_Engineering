// This #include statement was automatically added by the Particle IDE.
#include <Grove_ChainableLED.h>


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

// RGB led on D2
#define NUM_LEDS  2
ChainableLED leds(D2, D3, NUM_LEDS);
float hue_1 = 0.0;
float hue_2 = 0.0;

// FSR on A0
#define FSR A0

// VIBRO on A2
#define VIBRO A2

// HT on D4
#define DHTPIN 	D4
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

int currentTemp = 0;
int currentHum = 0;

// setup() runs once, when the device is first turned on
void setup() {
  // Put initialization like pinMode and begin functions here
  
// RGB led
leds.init();

// FSR
pinMode(FSR, INPUT);

// serial
Serial.begin(9600);

// VIBRO
pinMode(VIBRO, OUTPUT);

//HT
dht.begin();

}

// loop() runs over and over again, as quickly as it can execute.
void loop() {
  // The core of your code will likely live here.

  // Example: Publish event to cloud every 10 seconds. Uncomment the next 3 lines to try it!
  // Log.info("Sending Hello World to the cloud!");
  // Particle.publish("Hello world!");
  // delay( 10 * 1000 ); // milliseconds and blocking - see docs for more info!
  
  // RGB led
  
  float hue_1 = 0.8;
  float hue_2 = 0.3;

  
  leds.setColorHSB(0, hue_1, 1.0, 0.5);
  leds.setColorHSB(1, hue_2, 1.0, 0.5);
  
  // FSR
  int value = analogRead(FSR);
  Serial.println("The value of the FSR is:");
  Serial.println(value);
  
  digitalWrite(VIBRO, HIGH);
  delay(500);
  digitalWrite(VIBRO, LOW);
  delay(500);
  
  //HT
  currentTemp = (int)dht.getTempCelcius();
  currentHum = (int)dht.getHumidity();
  
  Serial.print("Humidity is: ");
  Serial.print(currentHum);
  Serial.println("%  ");
  delay(500);
  Serial.print("Temperature is: ");
  Serial.print(currentTemp);
  Serial.println("*C ");
  Serial.println();
  delay(500);
  
}



