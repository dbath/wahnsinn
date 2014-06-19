
// ENTER EXPERIMENT INFORMATION HERE:
int PRE_STIM_PERIOD = 5000;           // enter the time before first stimulus (milliseconds)
int STIM_BOUT_DURATION = 2000;           // enter the duration of stimulus bouts (milliseconds)
int STIM_FREQUENCY = 4;                  //enter the frequency of stimulus (Hertz)
int PULSE_WIDTH = 125;            //enter stimulus pulse width (milliseconds)
int RECOVER_BOUT_DURATION = 2000;    //enter the duration of inter-stimulus recovery periods (milliseconds)
int POST_STIM_PERIOD = 5000;          //enter the duration of post-stimulus observation (milliseconds)
int STIM_BOUT_NUM = 20;                // enter the number of stimulus bouts



//--------------------DO NOT CHANGE ANYTHING BELOW THIS LINE----------------------------------------------
int PERIOD = 1000/STIM_FREQUENCY;

const int stimPin1 =  0;      // stimulus 1: on (pulsed) during stimulus bouts on set 1
const int indPin1 =  1;      // indicator 1: on (constant) during stimulus bouts on set 1
const int progPin1 =  2;      // program 1: on when program is engaged on set 1
const int initPin1 =  3;      // initializer 1: high when button is pressed on set 1
const int stimPin2 =  4;      // stimulus 1: on (pulsed) during stimulus bouts on set 2
const int indPin2 =  5;      // indicator 1: on (constant) during stimulus bouts on set 2
const int progPin2 =  6;      // program 1: on when program is engaged on set 2
const int initPin2 =  7;      // initializer 1: high when button is pressed on set 2
const int stimPin3 =  8;      // stimulus 1: on (pulsed) during stimulus bouts on set 3
const int indPin3 =  9;      // indicator 1: on (constant) during stimulus bouts on set 3
const int progPin3 =  10;      // program 1: on when program is engaged on set 3
const int initPin3 =  11;      // initializer 1: high when button is pressed on set 3
const int test = 12;

//GLOBAL VARIABLES:
boolean lastReading1;  
boolean buttonState1 = LOW;
boolean lastButtonState1;
boolean lastReading2;  
boolean buttonState2 = LOW;
boolean lastButtonState2;
boolean lastReading3;  
boolean buttonState3 = LOW;
boolean lastButtonState3;
boolean run1 = false;
boolean run2 = false;
boolean run3 = false;


//------------------------------------------------------------------------------------------------------------

void setup()
{
  Serial.begin(9600);
  pinMode(stimPin1, OUTPUT);  
  pinMode(indPin1, OUTPUT); 
  pinMode(progPin1, OUTPUT); 
  pinMode(initPin1, INPUT);  
  pinMode(stimPin2, OUTPUT);  
  pinMode(indPin2, OUTPUT); 
  pinMode(progPin2, OUTPUT); 
  pinMode(initPin2, INPUT);    
  pinMode(stimPin3, OUTPUT);  
  pinMode(indPin3, OUTPUT); 
  pinMode(progPin3, OUTPUT); 
  pinMode(initPin3, INPUT); 
  pinMode(test, OUTPUT);
}



int programRun(int initTime, int initPin, int stimPin, int indPin, int progPin) {
  
  //unsigned long boutBegin = millis();
  int boutCount;
  unsigned long cycleTime;
  
  if (millis() < (initTime + ((STIM_BOUT_DURATION) + (RECOVER_BOUT_DURATION))*(STIM_BOUT_NUM) + PRE_STIM_PERIOD + POST_STIM_PERIOD)){
    digitalWrite(progPin, HIGH);  //illuminate program indicator LED.
  }
  else { digitalWrite(progPin, LOW); }
  
  if ((boutCount < STIM_BOUT_NUM) && (millis() >= (initTime + ((STIM_BOUT_DURATION) + (RECOVER_BOUT_DURATION))*(boutCount) + PRE_STIM_PERIOD))) {
    if (cycleTime < (initTime + ((STIM_BOUT_DURATION) + (RECOVER_BOUT_DURATION))*(boutCount) + PRE_STIM_PERIOD)) {
      unsigned long pulseBegin = millis();
      pulseTrain(stimPin, indPin, pulseBegin);
      boutCount += 1;
    }
    
  }    
  if(boutCount > STIM_BOUT_NUM){
    digitalWrite(progPin, LOW);
  }  

  cycleTime = millis();
}

int pulseTrain(int stimPin, int indPin, unsigned long pulseBegin) {
  int pulseCount = 0;
  int pulseState = 1;
  int prevPulseState = 1
  unsigned long pulseTime;
  
  if (millis() <= (pulseBegin + STIM_BOUT_DURATION) {
    digitalWrite(indPin, HIGH);
    if ((millis() < (pulseBegin + PULSE_WIDTH + (PERIOD)*(pulseCount))) && (millis() >= (pulseBegin + (PERIOD)*(pulseCount)))) {
      digitalWrite(stimPin, HIGH);
      pulseState = 1;
    }
    else { 
      digitalWrite(stimPin, LOW);
      pulseState = 0;
    }
    if (prevPulseState != pulseState) {
      if (pulseState = 0) {
        pulseCount += 1;
      }
      prevPulseState = pulseState;
    }   
  }
  else {
    digitalWrite(stimPin, LOW);
    digitalWrite(indPin, LOW);
  }
}

void loop() {
  unsigned long currentMillis = millis();
  int initPin;
  int stimPin;
  int indPin;
  int progPin;
  
  //READ BUTTON SET 1:
  int reading1 = digitalRead(initPin1);
  if (reading1 != lastReading1) {
    lastButtonState1 = reading1;
    if (reading1 == HIGH) {
      unsigned long initTime = currentMillis
      programRun(initTime, initPin1, stimPin1, indPin1, progPin1);
      //run1 = true;
    }
  }
  //if(run1 = true){
  //    programRun(currentMillis, initPin1, stimPin1, indPin1, progPin1);    CHECK THESE IN 2 & 3
  //}
    
  //READ BUTTON SET 2:
  int reading2 = digitalRead(initPin2);
  if (reading2 != lastReading2) {
    lastButtonState2 = reading2;
    if (reading2 == HIGH) {
      run2 = true;
    }
  } 
  if(run2=true) {
    programRun(currentMillis, initPin2, stimPin2, indPin2, progPin2);
  }
  
  //READ BUTTON SET 3:
  int reading3 = digitalRead(initPin3);
  if (reading3 != lastReading3) {
    lastButtonState3 = reading3;
    if (reading3 == HIGH) {
      run3 = true;
    }
  } 
  if(run3=true) {
    programRun(currentMillis, initPin3, stimPin3, indPin3, progPin3);
  }
}
