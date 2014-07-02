
// ENTER EXPERIMENT INFORMATION HERE:
unsigned long PRE_STIM_PERIOD = 60000;           // enter the time before first stimulus (milliseconds)
//unsigned long STIM_BOUT_DURATION = 2000;           // enter the duration of stimulus bouts (milliseconds)
unsigned long STIM_FREQUENCY = 5;                  //enter the frequency of stimulus (Hertz)
unsigned long PULSE_WIDTH = 100;            //enter stimulus pulse width (milliseconds)
//unsigned long RECOVER_BOUT_DURATION = 1000;    //enter the duration of inter-stimulus recovery periods (milliseconds)
unsigned long POST_STIM_PERIOD = 5000;          //enter the duration of post-stimulus observation (milliseconds)
//unsigned long STIM_BOUT_NUM = 2;                // enter the number of stimulus bouts



//--------------------DO NOT CHANGE ANYTHING BELOW THIS LINE----------------------------------------------

unsigned long PERIOD = 1000/STIM_FREQUENCY;
unsigned long OFF_PULSE = PERIOD - PULSE_WIDTH;
//unsigned long COMPAT_TEST = STIM_BOUT_DURATION % PERIOD;

const int stimPin =  5;      // stimulus: on (pulsed) during stimulus bouts
const int indPin =  2;      // indicator: on (constant) during stimulus bouts
const int progPin =  3;      // program: on when program is engaged
const int initPin =  6;      // initializer: high when button is pressed
const int piezoPin = 4;

//GLOBAL VARIABLES:
boolean lastReading;  
boolean buttonState = HIGH;
boolean lastButtonState = HIGH;


//------------------------------------------------------------------------------------------------------------

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
  delay(PRE_STIM_PERIOD);
  for(int x = 0; x < 3; x++) {
    pulseTrain(millis(), 10000);
    delay(10000);
  }
  for(int x = 0; x < 3; x++) {
    pulseTrain(millis(), 10000);
    delay(30000);
  }
  for(int x = 0; x < 3; x++) {
    pulseTrain(millis(), 20000);
    delay(60000);
  }
  
  delay(POST_STIM_PERIOD);
  offSound();
  digitalWrite(progPin, LOW);
}

void pulseTrain(unsigned long pulseBegin, int duration) {
  digitalWrite(indPin, HIGH);
  while ((millis() - pulseBegin) <= duration){
    digitalWrite(stimPin, HIGH);
    delay(PULSE_WIDTH);
    digitalWrite(stimPin, LOW);
    delay(OFF_PULSE);
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
  while (OFF_PULSE < 0) {
    errorMsg(1000);
  }
  
  //READ BUTTON:
  int reading = digitalRead(initPin);
  if (reading != lastButtonState) {
    if (reading == HIGH) {
      programRun(millis());
    }
    lastButtonState = reading;
  }
}
