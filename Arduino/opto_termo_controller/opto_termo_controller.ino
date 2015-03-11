#include <Scheduler.h>


const int initPin =  13;      // initializer: high when button is pressed
const int bluePin = 12;
const int redPin = 11;
const int greenPin = 10;
const int stimPin =  redPin;      // stimulus: on (pulsed) during stimulus bouts

const int indPin =  9;      // indicator: on (constant) during stimulus bouts
const int progPin =  8;      // program: on when program is engaged
const int piezoPin = 7;

const int thermocouple = A0; //ANALOG INPUT PIN;
const int heaterPin = 3;
const int coolerPin = 2;
double starttime = millis();
boolean buttonPressed = false;
double setTemperature = 23.0;
double currentTemperature = 0.0;
double tempUpDelta = 0.0;
double tempDnDelta = 0.0;
double peakThreshold = 50.0;


//GLOBAL VARIABLES:

boolean lastReading;  
boolean buttonState = HIGH;
boolean lastButtonState = HIGH;
boolean Rising = true;
const int analogBitRate = 16;

void setup(){
  Serial.begin(9600);
  analogReadResolution(analogBitRate); 
  pinMode(stimPin, OUTPUT);  
  pinMode(indPin, OUTPUT); 
  pinMode(progPin, OUTPUT); 
  pinMode(initPin, INPUT); 
  pinMode(piezoPin, OUTPUT);
  pinMode(thermocouple, INPUT);
  pinMode(heaterPin, OUTPUT);
  pinMode(coolerPin, OUTPUT);
  Scheduler.startLoop(buttonReader);
  analogWrite(stimPin, 0);
  analogWrite(indPin, 0);
}

void programRun(unsigned long initTime) {
  
  onSound();
  analogWrite(progPin, 200);
  buttonPressed = false; 
  //------------------------------------------------------------------------------------ ENTER PROGRAM INFORMATION BELOW THIS LINE-----------------------------------------------
  
  
  //  -----NOTES ABOUT runBlock-----
  //  describe each block of stimuli with:
       // number of repeats (integer)
       // bout duration (integer, in milliseconds)
       // recovery time between bouts (integer, in milliseconds)
       // pulse frequency (integer, in Hz)
       // pulse width (integer, in milliseconds)
       // intensity (integer, precent of max).
  // example : runBlock(3, 10000, 10000, 100, 5, 128)  is 3 bouts of 10 second stimuli (5ms pulses, 100Hz, 50% intensity), with 10 seconds recovery between bouts.
  
  // NOTES ABOUT rampTemp:
     // define a change in temperature using three parameters:
       // rate of temperature increase (positive number, in degrees per minute)
       // peak temperature at which to stop increasing ( positive number, in degrees celsius)
       // rate of temperature decrease (positive number, in degrees per minute)
       
       
  delay(1000);  // enter the pre-stimulus period here, in ms. (ex ""   delay(30000); "" )

  rampTemp(1.0, 28.0, 1.0);   //this sets the temperature protocol. see notes above.
  runBlock(100, 1000, 1000, 20, 25, 25); //this initiates a light stimulus. lines below will not be read until this Block finishes.
  // copy and past rampTemp or runBlock calls ad libitum here.
  delay(1000);    // enter the post-stimulus period here, in ms. (ex ""   delay(30000); "" )
  
  //------------------------------------------------------------------------------------ENTER PROGRAM INFORMATION ABOVE THIS LINE-----------------------------------------------------------  

  offSound();
  analogWrite(progPin, 0);
  buttonPressed=false;

}

void runBlock(int repeats, unsigned long duration, unsigned long recovery, long frequency, long pulse_width, int pwm) {
  for(int x = 0; x < repeats; x++) {
    pulseTrain(millis(), duration, frequency, pulse_width, pwm);
    delay(recovery);
  }
}

void pulseTrain(unsigned long pulseBegin, unsigned long duration, long frequency, long pulse_width, int pwm) {
  analogWrite(indPin, 255);
  while ((millis() - pulseBegin) <= duration){
    analogWrite(stimPin, pwm*2.55);
    delay(pulse_width);
    analogWrite(stimPin, 0);
    delay((1000L/frequency) - pulse_width);
  }
  analogWrite(indPin, 0);
}  


void onSound() {
  tone(piezoPin, 440, 250);
}

void offSound() {
  tone(piezoPin, 880, 250);
}

void errorMsg(int time) {
  analogWrite(progPin, 0);
  analogWrite(indPin, 255);
  analogWrite(stimPin, 0);
  delay(time/2);
  analogWrite(progPin, 255);
  analogWrite(indPin, 0);
  delay(time/2);
}
  
void rampTemp(unsigned long upRate, unsigned long peakTemp, unsigned long downRate) {
  Rising = true;
  tempUpDelta = upRate / 60.0;
  tempDnDelta = downRate / -60.0;
  peakThreshold = peakTemp;
}

void loop() {
  
  int val = analogRead(thermocouple);
  double Voltage = val * (3.35 / (pow(2.0, analogBitRate)-1.0)) ;
  double currentTemperature = (Voltage - 1.25)/0.005; //calibrated to ice bath, D. Bath 150220.
  double t = (millis() - starttime)/60000;
  Serial.print(t);
  Serial.print("min    Temperature:  ");
  Serial.print(currentTemperature);
  
  
  if (currentTemperature >= peakThreshold) {
    tempUpDelta = 0.0;
    Rising = false;
  }
  if (Rising == true) {
    setTemperature = setTemperature + tempUpDelta;
    Serial.print(" and rising at: ");
    Serial.print(tempUpDelta);
    Serial.print("deg/sec ");  
    Serial.print(setTemperature);
  }
  if (Rising == false) {
    setTemperature = setTemperature + tempDnDelta;
    Serial.print(" and falling at: ");
    Serial.print(tempDnDelta);
    Serial.print("deg/sec ");
  }
  if (setTemperature - currentTemperature <=0.1) {
    Serial.print(" heater: OFF");
    Serial.println(buttonPressed);
    digitalWrite(heaterPin, LOW);
    digitalWrite(coolerPin, HIGH);
  }
  else if (setTemperature - currentTemperature >=0.1){
    
    Serial.print(" heater: ON");
    Serial.println(buttonPressed);
    digitalWrite(coolerPin, LOW);
    digitalWrite(heaterPin, HIGH);
  }
  else {
    digitalWrite(heaterPin, LOW);
    digitalWrite(coolerPin, LOW);
  }

  delay(1000);
}


  
void buttonReader() {
  //READ BUTTON:
    if (setTemperature - currentTemperature >=0.1) {
      if (buttonPressed == true) {
        Serial.println("**************initiating program********************");
        programRun(millis());
      }
    }
    
    int reading = digitalRead(initPin);
    if (reading != lastButtonState) {
      lastButtonState = reading;
      if (reading == HIGH) {
        buttonPressed = true;
        Serial.print(buttonPressed);
      }
    }
    

    
    yield();
  }




/*
 Tone generator
 v1  use timer, and toggle any digital pin in ISR
   funky duration from arduino version
   TODO use FindMckDivisor?
   timer selected will preclude using associated pins for PWM etc.
    could also do timer/pwm hardware toggle where caller controls duration
*/


// timers TC0 TC1 TC2   channels 0-2 ids 0-2  3-5  6-8     AB 0 1
// use TC1 channel 0 
#define TONE_TIMER TC1
#define TONE_CHNL 0
#define TONE_IRQ TC3_IRQn

// TIMER_CLOCK4   84MHz/128 with 16 bit counter give 10 Hz to 656KHz
//  piano 27Hz to 4KHz

static uint8_t pinEnabled[PINS_COUNT];
static uint8_t TCChanEnabled = 0;
static boolean pin_state = false ;
static Tc *chTC = TONE_TIMER;
static uint32_t chNo = TONE_CHNL;

volatile static int32_t toggle_count;
static uint32_t tone_pin;

// frequency (in hertz) and duration (in milliseconds).

void tone(uint32_t ulPin, uint32_t frequency, int32_t duration)
{
		const uint32_t rc = VARIANT_MCK / 256 / frequency; 
		tone_pin = ulPin;
		toggle_count = 0;  // strange  wipe out previous duration
		if (duration > 0 ) toggle_count = 2 * frequency * duration / 1000;
		 else toggle_count = -1;

		if (!TCChanEnabled) {
 			pmc_set_writeprotect(false);
			pmc_enable_periph_clk((uint32_t)TONE_IRQ);
			TC_Configure(chTC, chNo,
				TC_CMR_TCCLKS_TIMER_CLOCK4 |
				TC_CMR_WAVE |         // Waveform mode
				TC_CMR_WAVSEL_UP_RC ); // Counter running up and reset when equals to RC
	
			chTC->TC_CHANNEL[chNo].TC_IER=TC_IER_CPCS;  // RC compare interrupt
			chTC->TC_CHANNEL[chNo].TC_IDR=~TC_IER_CPCS;
			 NVIC_EnableIRQ(TONE_IRQ);
                         TCChanEnabled = 1;
		}
		if (!pinEnabled[ulPin]) {
			pinMode(ulPin, OUTPUT);
			pinEnabled[ulPin] = 1;
		}
		TC_Stop(chTC, chNo);
                TC_SetRC(chTC, chNo, rc);    // set frequency
		TC_Start(chTC, chNo);
}

void noTone(uint32_t ulPin)
{
	TC_Stop(chTC, chNo);  // stop timer
	digitalWrite(ulPin,LOW);  // no signal on pin
}

// timer ISR  TC1 ch 0
void TC3_Handler ( void ) {
	TC_GetStatus(TC1, 0);
	if (toggle_count != 0){
		// toggle pin  TODO  better
		digitalWrite(tone_pin,pin_state= !pin_state);
		if (toggle_count > 0) toggle_count--;
	} else {
		noTone(tone_pin);
	}
}

