#include <SoftwareSerial.h>                // Serial 통신을 하기 위해 선언

SoftwareSerial BTSerial(7, 8);             // HC-06모듈 7=TXD , 8=RXD 핀 선언 

void setup(){
    Serial.begin(9600);                          // 시리얼 통신 선언 (보드레이트 9600)
    BTSerial.begin(9600);                     // HC-06 모듈 통신 선언 (보드레이트 9600)
}

void loop(){
    if(BTSerial.available())                     // BTSerial에 입력이 되면
      Serial.write(BTSerial.read());           // BTSerial에 입력된 값을 시리얼 모니터에 출력
    if(Serial.available())                          // 시리얼 모니터에 입력이 되면
    BTSerial.write(Serial.read());           // 그 값을 BTSerial에 출력

}
[출처] 아두이노 HC-06 블루투스 2.0 모듈 사용 예제|작성자 도매키트 아두몰
