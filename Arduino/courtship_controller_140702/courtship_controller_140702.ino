
const int stimPin =  5;      // stimulus: on (pulsed) during stimulus bouts
const int indPin =  2;      // indicator: on (constant) during stimulus bouts
const int progPin =  3;      // program: on when program is engaged
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
  
  
  delay(30000);  // enter the pre-stimulus period here, in ms. (ex ""   delay(30000); "" )
  
  //  describe each block of stimuli with number of repeats, bout duration, recovery time between bouts, pulse frequency, and pulse width.
  // example : runBlock(3, 10000, 10000, 100, 5)  is 3 bouts of 10 second stimuli (5ms pulses, 100Hz), with 10 seconds recovery between bouts.
  runBlock(3, 1000, 5000, 100, 5);
  runBlock(3, 5000, 10000, 20, 25);
  runBlock(3, 10000, 20000, 5, 100);
  runBlock(3, 30000, 30000, 5, 100);
  
  delay(30000);    // enter the post-stimulus period here, in ms. (ex ""   delay(30000); "" )
  
  //------------------------------------------------------------------------------------ENTER PROGRAM INFORMATION ABOVE THIS LINE-----------------------------------------------------------  



}

void runBlock(unsigned long millis(), int repeats, int duration, int recovery, int frequency, int pulse_width) {
  for(int x = 0; x < repeats; x++) {
    pulseTrain(millis(), duration);
    delay(recovery);
  }
}

void pulseTrain(unsigned long pulseBegin, int duration, int frequency, int pulse_width) {
  digitalWrite(indPin, HIGH);
  while ((millis() - pulseBegin) <= duration){
    digitalWrite(stimPin, HIGH);
    delay(pulse_width);
    digitalWrite(stimPin, LOW);
    delay((1000/frequency) - pulse_width);
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
