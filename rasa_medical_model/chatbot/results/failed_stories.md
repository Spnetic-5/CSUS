## happy path 1 (C:\Users\Utsav\AppData\Local\Temp\tmp1iorbp2n\96ea7e3931e24f7f9bacec026b69070c_conversation_tests.md)
* greet: hello there!
    - utter_greet
* mood_great: amazing   <!-- predicted: smalltalk_appraisal_good: amazing -->
    - utter_happy   <!-- predicted: utter_smalltalk_appraisal_good -->


## happy path 2 (C:\Users\Utsav\AppData\Local\Temp\tmp1iorbp2n\96ea7e3931e24f7f9bacec026b69070c_conversation_tests.md)
* greet: hello there!
    - utter_greet
* mood_great: amazing   <!-- predicted: smalltalk_appraisal_good: amazing -->
    - utter_happy   <!-- predicted: utter_smalltalk_appraisal_good -->
* goodbye: bye-bye!
    - utter_goodbye


## sad path 1 (C:\Users\Utsav\AppData\Local\Temp\tmp1iorbp2n\96ea7e3931e24f7f9bacec026b69070c_conversation_tests.md)
* greet: hello
    - utter_greet
* mood_unhappy: not good   <!-- predicted: smalltalk_appraisal_bad: not good -->
    - utter_cheer_up   <!-- predicted: utter_smalltalk_appraisal_bad -->
    - utter_did_that_help   <!-- predicted: action_listen -->
* affirm: yes
    - utter_happy   <!-- predicted: utter_ask_howdoing -->


## sad path 2 (C:\Users\Utsav\AppData\Local\Temp\tmp1iorbp2n\96ea7e3931e24f7f9bacec026b69070c_conversation_tests.md)
* greet: hello
    - utter_greet
* mood_unhappy: not good   <!-- predicted: smalltalk_appraisal_bad: not good -->
    - utter_cheer_up   <!-- predicted: utter_smalltalk_appraisal_bad -->
    - utter_did_that_help   <!-- predicted: action_listen -->
* deny: not really
    - utter_goodbye


## sad path 3 (C:\Users\Utsav\AppData\Local\Temp\tmp1iorbp2n\96ea7e3931e24f7f9bacec026b69070c_conversation_tests.md)
* greet: hi
    - utter_greet
* mood_unhappy: very terrible   <!-- predicted: smalltalk_appraisal_bad: very terrible -->
    - utter_cheer_up   <!-- predicted: utter_smalltalk_appraisal_bad -->
    - utter_did_that_help   <!-- predicted: action_listen -->
* deny: no
    - utter_goodbye


## bot challenge (C:\Users\Utsav\AppData\Local\Temp\tmp1iorbp2n\96ea7e3931e24f7f9bacec026b69070c_conversation_tests.md)
* bot_challenge: are you a bot?   <!-- predicted: smalltalk_agent_chatbot: are you a bot? -->
    - utter_iamabot   <!-- predicted: utter_smalltalk_agent_chatbot -->


