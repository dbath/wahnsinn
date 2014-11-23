
const int stimPin =  5;      // stimulus: on (pulsed) during stimulus bouts
const int indPin =  3;      // indicator: on (constant) during stimulus bouts
const int progPin =  2;      // program: on when program is engaged
const int initPin =  6;      // initializer: high when button is pressed
const int piezoPin = 4;

//GLOBAL VARIABLES:
boolean lastReading;  
boolean buttonState = HIGH;
boolean lastButtonState = HIGH;

void setup()
{
  Serial.begin(9600);
  pinMode(stimPin, OUTPUT);  
  pinMode(indPin, OUTPUT); 
  pinMode(progPin, OUTPUT); 
  pinMode(initPin, INPUT); 
  pinMode(piezoPin, OUTPUT);
}

void programRun(unsigned long initTime) {
  
  onSound();
  digitalWrite(progPin, HIGH);
  
  
  
  
  //------------------------------------------------------------------------------------ ENTER PROGRAM INFORMATION BELOW THIS LINE-----------------------------------------------
  
  
  delay(120000);  // enter the pre-stimulus period here, in ms. (ex ""   delay(30000); "" )
  
  //  describe each block of stimuli with number of repeats, bout duration, recovery time between bouts, pulse frequency, pulse width and intensity (integer, precent of max).
  // example : runBlock(3, 10000, 10000, 100, 5, 128)  is 3 bouts of 10 second stimuli (5ms pulses, 100Hz, 50% intensity), with 10 seconds recovery between bouts.
  runBlock(2, 60000, 180000, 20, 25, 50);
  
  delay(60000);    // enter the post-stimulus period here, in ms. (ex ""   delay(30000); "" )
  
  //------------------------------------------------------------------------------------ENTER PROGRAM INFORMATION ABOVE THIS LINE-----------------------------------------------------------  

  offSound();
  digitalWrite(progPin, LOW);

}

void runBlock(int repeats, unsigned long duration, unsigned long recovery, long frequency, long pulse_width, int pwm) {
  for(int x = 0; x < repeats; x++) {
    pulseTrain(millis(), duration, frequency, pulse_width, pwm);
    delay(recovery);
  }
}

void pulseTrain(unsigned long pulseBegin, unsigned long duration, long frequency, long pulse_width, int pwm) {
  digitalWrite(indPin, HIGH);
  while ((millis() - pulseBegin) <= duration){
    analogWrite(stimPin, pwm*2.55);
    delay(pulse_width);
    digitalWrite(stimPin, LOW);
    delay((1000L/frequency) - pulse_width);
  }
  digitalWrite(indPin, LOW);
}  


void onSound() {
  tone(piezoPin, 440, 250);
}

void offSound() {
  tone(piezoPin, 880, 250);
}

void errorMsg(int time) {
  digitalWrite(progPin, LOW);
  digitalWrite(indPin, HIGH);
  digitalWrite(stimPin, LOW);
  delay(time/2);
  digitalWrite(progPin, LOW);
  digitalWrite(indPin, HIGH);
  delay(time/2);
}
  
void loop() {
  
  //READ BUTTON:
  int reading = digitalRead(initPin);
  if (reading != lastButtonState) {
    if (reading == HIGH) {
      programRun(millis());
    }
    lastButtonState = reading;
  }
}
