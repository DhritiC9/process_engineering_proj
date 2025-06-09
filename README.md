# 1. T5 Model Testing

I started by testing the T5-small model to understand how it works. For this, I implemented a basic T5 setup and also used a code example from GitHub related to tag prediction using the T5 model. These can be found in:
	•	T5_basecode_test.py
	•	T5small_Tag_Finetune_test.ipynb


# 2. T5_PFD-PID.ipynb

Next, I wrote a basic script to convert PFD strings to PID strings based on the dataset provided in the original research paper.

Dataset Used (same across all files):
	•	Training: train_data_10k.json
	•	Testing: test_data_1k.json
	•	Evaluation: eval_data_1k.json

Model: T5-small

Training Parameters:
``` 
output_dir = "./t5_pid_model"  
evaluation_strategy = "steps"  
eval_steps = 500  
logging_steps = 100  
save_steps = 500  
save_total_limit = 2  
learning_rate = 5e-5  
per_device_train_batch_size = 16  
per_device_eval_batch_size = 16  
num_train_epochs = 3  
weight_decay = 0.01  
load_best_model_at_end = True  
report_to = "none"  # Change to "tensorboard" if logging is needed  
fp16 = torch.cuda.is_available()  
``` 

  
Inference Output:

Input PFD:
``` 
(raw)(hex){1}(hex){2}(mix)<2(r)[{tout}(v)(prod)]{bout}(v)(splt)[(hex){2}(hex){3}(pp)(v)(mix)<1(r)[{bout}(v)(prod)]{tout}(v)(splt)[(hex){4}(r)[{tout}(v)(prod)]{bout}(v)(hex){4}(prod)](v)1](v)2n|(raw)(hex){1}(v)(prod)n|(raw)(hex){3}(v)(prod)
``` 
Predicted PID:
``` 
(raw)(hex)1(C)TC_1(hex)2(C)TC_2(mix)2(r)_3[(C)TC_3][(C)LC_4][tout(C)PC_5(v)_5(prod)]bout(v)_5(splt)[(hex)2(hex)3(C)TC_7(pp)[(C)M](C)PI(C)FC_8(v)_8(mix)1(r)_9[(C)TC_9][(C)LC_10][bout(v)_10(prod)]tout(C)PC_11(v)_11(splt)[(hex)2(C)TC_12(pp)[(C)M
``` 
Ground Truth PID:
``` 
(raw)(hex){1}(C){TC}_1(hex){2}(mix)<2(r)<_2[(C){TC}_2][(C){LC}_3][{tout}(C){PC}_4(v)<_4(prod)]{bout}(v)<_3(splt)[(hex){2}(hex){3}(C){TC}_5(pp)[(C){M}](C){PI}(C){FC}_6(v)<_6(mix)<1(r)<_7[(C){TC}_7][(C){LC}_8][{bout}(v)<_8(prod)]{tout}(C){PC}_9(v)<_9(splt)[(C){FC}_10(v)1<_10](hex){4}(r)<_11[(C){TC}_11][(C){LC}_12][{tout}(C){PC}_13(v)<_13(prod)]{bout}(v)<_12(hex){4}(prod)](C){FC}_14(v)2<_14n|(raw)(hex){1}(v)<_1(prod)n|(raw)(hex){3}(v)<_5(prod)
``` 
We can clearly observe that:
	•	The brackets are mismatched and do not balance.
	•	The < direction symbols, which are crucial to indicating process flow connections, are missing.
	•	The output structure and length are far from the expected PID format.

So, in the next step, I aimed to address these issues and improve the prediction quality.



# 3. Copy_of_T5_PFDtoPID.ipynb

I reran the same model (T5-small) on the same dataset, but this time I improved some of the training parameters, primarily increasing the number of epochs and adjusting the step intervals.

Updated Training Parameters:
``` 
training_args = TrainingArguments(
    output_dir = "./t5_pid_model",
    eval_strategy = "steps",
    eval_steps = 20,
    logging_steps = 50,
    save_steps = 40,
    save_total_limit = 2,
    learning_rate = 5e-5,
    remove_unused_columns = False,
    per_device_train_batch_size = 32,
    per_device_eval_batch_size = 16,
    num_train_epochs = 10,
    weight_decay = 0.01,
    load_best_model_at_end = True,
    report_to = "tensorboard",  # Can change to "none" if logging not required
    fp16 = torch.cuda.is_available(),
)
``` 
Inference Output:

Input PFD:
``` 
(raw)(hex){1}(comp)(v)(mix)<&|(raw)(v)&|(mix)<&|(raw)(mix)<2(pp)(splt)[(v)2](v)&|(hex){2}(mix)<1(pp)(splt)[(r)(v)(hex){2}(prod)](v)1n|(raw)(hex){1}(v)(prod) ...
``` 
Predicted PID:
``` 
(raw)(hex)1(comp)(v)(mix)&|(raw)(mix)&|(raw)(mix)1(pp)(splt)[(r)(v)(hex)2(prod)](v)1n|(raw)(hex)1(v)(prod)](v)1n|(raw)(hex)1(prod)(pro
``` 
True PID:
``` 
(raw)(hex){1}(C){TC}_1(hex){2}(mix)<2(r)<_2[(C){TC}_2][(C){LC}_3][{tout}(C){PC}_4(v)<_4(prod)]{bout}(v)<_3(splt)[(hex){2}(hex){3}(C){TC}_5(pp)[(C){M}](C){PI}(C){FC}_6(v)<_6(mix)<1(r)<_7[(C){TC}_7][(C){LC}_8][{bout}(v)<_8(prod)]{tout}(C){PC}_9(v)<_9(splt)[(C){FC}_10(v)1<_10](hex){4}(r)<_11[(C){TC}_11][(C){LC}_12][{tout}(C){PC}_13(v)<_13(prod)]{bout}(v)<_12(hex){4}(prod)(C{FC}_14(v)2<_14n|(raw)(hex){1}(v)<_1(prod)n|(raw)(hex){3}(v)<_5(prod)
``` 
Despite these changes, the issues persisted:
	•	Bracket structure is incorrect.
	•	Flow directions (<) are missing.
	•	Output length and structure are wrong.

So, next I tried : -



# 4. T5_small.ipynb

Next, I tested the same model — T5-small — again, but with different parameters

``` 
training_args = TrainingArguments(
    output_dir = "./t5_pid_model",
    eval_strategy = "steps",
    eval_steps = 200,
    logging_steps = 100,
    save_steps = 200,
    save_total_limit = 2,
    learning_rate = 5e-5,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    num_train_epochs = 3,
    weight_decay = 0.01,
    load_best_model_at_end = True,
    metric_for_best_model = "eval_loss",
    greater_is_better = False,
    report_to = "tensorboard",
    fp16 = torch.cuda.is_available(),
)
``` 
Inference Output:

Input PFD:
``` 
(raw)(hex){1}(v)(mix)<&|(raw)(v)(mix)&<&|(raw)&||(r)[{bout}(v)(prod)]{tout}(v)(pp)(v)(prod)n|(raw)(splt)[(hex){1}(mix)<1(prod)](v)1
``` 
Predicted PID:
``` 
(raw)(hex)1(v)(mix)&|(raw)[(raw)(splt)[(raw)(splt)[bout(v)1(mix)&||(raw)tout(v)(prod)][(raw)[tout(v)(v)1(mix)&|(raw)(v)tout(v)
``` 
True PID:
``` 
(raw)(hex){1}(C){TC}_1(C){FC}_2(v)<_2(mix)<&|(raw)(C){FFC}_3<_4(v)<_3(mix)&<&|(raw)(C){FT}&_4||(r)[(C){TI}][(C){LC}_5][{bout}(v)<_5(prod)]{tout}(C){PC}_6(v)<_6(pp)[(C){M}](C){PI}(C){FC}_7(v)<_7(prod)n|(raw)(splt)[(hex){1}(mix)<1(prod)](v)1<_1
``` 
Conclusion:

Again, the output is clearly off




# 5. Alternative Technique

Most of the previous issues were due to tokenisation.

Large Language Models (LLMs) like T5 and Gemini are typically trained to handle real-world text and natural language, not symbolic data or sequences of special characters. During tokenization, these models convert the text into tokens, but in doing so, many important symbols such as brackets and arrows were getting lost. This was the reason for incorrect predictions and mismatched structures.

To work around this, I attempted to replace all special characters with corresponding alphabetic representations, for example:
``` 
 '[' = OB (Open Bracket)  
 and '}' = CB (Close Bracket)
``` 

This way, the model would be working with alphabetic tokens (closer to real “words”) instead of symbols.

But unfortunately, this did not work as expected.
The model began generating incorrect and inconsistent representations such as:
``` 
OBB or BOB
``` 
These outputs were not reversible and couldn't be reversed back to the original symbols.


The original research paper suggested to pretrain the T5 model from scratch on the given dataset. Their idea was based on the fact that no LLM models exist that are capable of handling symbolic or character-based data.

They first pretrained the T5 model using their own structured dataset, and then fine-tuned it for the PFD-to-PID conversion task.



Before concluding to having to finetune a model of our own I next experimented with the BART model.



# 6. BART Model

After facing issues with T5 and tokenization problems, I decided to use the BART model to see if it could handle symbolic data better. BART performed somewhat better — especially in handling brackets and the (<, >) symbols correctly — but the predictions were still structurally incorrect.


## 6.1 Bart_PFDtoPID.ipynb
``` 
training_args = TrainingArguments(
    output_dir="./bart_pid_model",
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=100,
    save_steps=100,
    save_total_limit=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="tensorboard",
    fp16=torch.cuda.is_available()
)
``` 
Example Output:

Input:
``` 
(raw)(hex){1}(mix)<&|(raw)(v)&|(v)(hex){2}(rect)<1<2[{tout}(cond)(sev)[(v)(prod)](splt)[(v)(prod)](v)1|{bout}(splt)[(v)(prod)](hex){3}2n|(raw)(hex){1}(v)(prod)n|(raw)(v)(hex){2}(prod)n|(raw)(v)(hex){3}(prod)]
``` 
Predicted:
``` 
(raw)(hex){1}(mix)<&|(raw(v)&|(-v)(hex)-2}(hex){3}(rect)<1<2[{tout}(cond)(sev)[(v)(prod)](splt)<2<3[{bout}-cond)(v)(h)<3<3(v)-prod](splint)[(V)(proder)](v(v))1|{bouts}(spltt)[(vo)(proda)](hex)3}2n|(sing)(hex)=1}{(v-prod)n| (raw)(v)hex){2}{(prod)(hex)}(hex)|(hex)]
``` 
Although BART preserved some structural tokens, it still made up unknown tokens like splint, proder, proda, etc.


## 6.2 Bart_PFDtoPID_2.ipynb
``` 
training_args = TrainingArguments(
    output_dir="./bart_pid_model",
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=100,
    save_steps=100,
    save_total_limit=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    num_train_epochs=10,
    weight_decay=0.01,
    report_to="tensorboard",
    fp16=torch.cuda.is_available()
)
``` 
Example Input:
``` 
(raw)(pp)(v)(v)(mix)<&|(raw)(hex)(v)&|(mix)<&|(raw)(v)&|(pp)(v)(mix)<1(r)(v)(splt)[(prod)](v)1
``` 
Prediction:
``` 
(raw)(hex)(C){TC}_1(C){FC}_2(v)&<_2|(mix)<1(r)<_3<&|(raw(pp)[(C)M}](C){PI}(C(){FC}}_3(v]<_4(C {FC}&_5|[(C)[TC}][(C)(LC}_6][{tout}(raw)](v)(splt)[(prod)](CZI}_7<_7(v)]1<_8
``` 
True Output:
``` 
(raw)(hex)<_1(C){TC}_1(C){FC}_2(v)<_2(mix)<&|(raw)(pp)[(C){M}](C){PI}(C){FC}_3(v)<_3(C){FC}_4(v)&<_4|(mix)<&|(raw)(C){FC}_5(v)&<_5|(pp)[(C){M}](C){PI}(C){FC}_6(v)<_6(mix)<1(r)<_7[(C){TC}_7][(C){LC}_8](v)<_8(splt)[(prod)](C){FC}_9(v)1<_9
``` 
BART got closer to the correct output format but still introduced malformed brackets, and made up structures (like CZI).


## 6.3 Bart_PFDtoPID_3.ipynb

Reused the same training arguments as above for another round of training on the same architecture. The output remained similar in structure and issues


Despite improved bracket preservation, BART failed to generate syntactically and semantically valid PID outputs. 



Since even BART wasn’t yielding usable results, the next step was to experiment with ByT5.



# 7. byt5-3.ipynb

The model used is ByT5, a variant of the T5 transformer that processes text at the character (byte) level instead of the word or subword level. This makes ByT5 suited for our task, which involves symbolic strings.

I fine-tuned ByT5 for just 2 epochs due to limited GPU availability and time constraints. Despite only using 2 epochs, the model produced the most accurate and structured outputs so far, indicating strong potential for learning such symbol-heavy sequences.

With additional training time, GPU and Epochs we can reasonably expect even better performance and structural consistency.


RESULTS :- 

   PFD Input 2:
   ``` 
(raw)(v)(tank)(pp)(v)(r)<1[{bout}(v)(prod)]{tout}(v)(splt)[(hex){1}(prod)](v)(mix)1<&|(raw)(hex){2}&|n|(raw)(hex){2}(v)(prod)n|(raw)(hex){1}(v)(prod)
``` 
 Prediction A:
 ``` 
(raw)(hex){1}(C){TC}_1(mix)<1(r)<_1<&|(raw)(v)<_2(tank)[(C){LC}_2](pp)[(C){M}](C){PI}(C){FC}_3(v)<_3(C){FC}_4(v)&<_4|[(C){TC}_5][(C){LC}_6][{bout}(v)<_6(prod)]{tout}(C){PC}_7(v)<_7(splt)[(hex){2}(C){TC}_8(prod)](C){FC}_9(v)1<_9n|(raw)(hex){2}(v)<_1(prod)n|(raw)(hex){1}(v)<_1(prod)
``` 
 Prediction B:
 ``` 
(raw)(hex){1}(C){TC}_1(mix)<1(r)<_1<&|(raw)(v)<_2(tank)[(C){LC}_2](pp)[(C){M}](C){PI}(C){FC}_3(v)<_3(C){FC}_4(v)&<_4|[(C){TC}_5][(C){LC}_6][{bout}(v)<_6(prod)]{tout}(C){PC}_7(v)<_7(splt)[(hex){2}(C){TC}_8(prod)](C){FC}_9(v)1<_9n|(raw)(hex){2}(v)<_3(prod)n|(raw)(hex){1}(v)<_1(prod)
``` 
 Prediction C:
 ``` 
(raw)(hex){1}(C){TC}_1(mix)<1(r)<_1<&|(raw)(v)<_2(tank)[(C){LC}_2](pp)[(C){M}](C){PI}(C){FC}_3(v)<_3(C){FC}_4(v)&<_4|[(C){TC}_5][(C){LC}_6][{bout}(v)<_6(prod)]{tout}(C){PC}_7(v)<_7(splt)[(hex){2}(C){TC}_8(prod)](C){FC}_9(v)1<_9n|(raw)(hex){2}(v)<_5(prod)n|(raw)(hex){1}(v)<_1(prod)
``` 
 Prediction D:
 ``` 
(raw)(hex){1}(C){TC}_1(mix)<1(r)<_1<&|(raw)(v)<_2(tank)[(C){LC}_2](pp)[(C){M}](C){PI}(C){FC}_3(v)<_3(C){FC}_4(v)&<_4|[(C){TC}_5][(C){LC}_6][{bout}(v)<_6(prod)]{tout}(C){PC}_7(v)<_7(splt)[(hex){2}(C){TC}_8(prod)](C){FC}_9(v)1<_9n|(raw)(hex){2}(v)<_4(prod)n|(raw)(hex){1}(v)<_1(prod)
``` 
 Prediction E:
 ``` 
(raw)(hex){1}(C){TC}_1(mix)<1(r)<_1<&|(raw)(v)<_2(tank)[(C){LC}_2](pp)[(C){M}](C){PI}(C){FC}_3(v)<_3(C){FC}_4(v)&<_4|[(C){TC}_5][(C){LC}_6][{bout}(v)<_6(prod)]{tout}(C){PC}_7(v)<_7(splt)[(hex){2}(C){TC}_8(prod)](C){FC}_9(v)1<_9n|(raw)(hex){2}(v)<_6(prod)n|(raw)(hex){1}(v)<_1(prod)
``` 
 Ground Truth PID:
 ``` 
(raw)(v)<_1(tank)[(C){LC}_1](pp)[(C){M}<_2](C){PI}(C){FC}_2(C){FC}_3(v)<_3(r)<1<_4[(C){TC}_4][(C){LC}_5][{bout}(v)<_5(prod)]{tout}(C){PC}_6(v)<_6(splt)[(hex){1}(C){TC}_7(prod)](C){FC}_8(v)<_8(mix)1<&|(raw)(hex){2}(C){TC}&_9|n|(raw)(hex){2}(v)<_9(prod)n|(raw)(hex){1}(v)<_7(prod)
```
===================================================

 PFD Input 3:
``` 
(raw)(pp)(v)(v)(r)<&|(raw)(hex){1}(comp)&|[{tout}(v)(prod)]{bout}(v)(hex){2}(prod)n|(raw)(v)(hex){1}(prod)n|(raw)(hex){2}(v)(prod)
``` 
 Prediction A:
 ``` 
(raw)(hex){1}(C){TC}_1(comp)[(C){M}<_2](C){PC}_2(r)<_5<&|(raw)(pp)[(C){M}](C){PI}(C){FC}_3(v)<_3(C){FC}_4(v)&<_4|[(C){TC}_5][(C){LC}_6][{tout}(C){PC}_7(v)<_7(prod)]{bout}(v)<_6(hex){2}(C){TC}_8(prod)n|(raw)(C){FC}_9<_1(v)<_9(hex){1}(prod)n|(raw)(hex){2}(v)<_10(prod)
``` 
 Prediction B:
 ``` 
(raw)(hex){1}(C){TC}_1(comp)[(C){M}<_2](C){PC}_2(r)<_8<&|(raw)(pp)[(C){M}](C){PI}(C){FC}_3(v)<_3(C){FC}_4(v)&<_4|[(C){TC}_5][(C){LC}_6][{tout}(C){PC}_7(v)<_7(prod)]{bout}(v)<_6(hex){2}(C){TC}_8(prod)n|(raw)(C){FC}_9<_1(v)<_9(hex){1}(prod)n|(raw)(hex){2}(v)<_10(prod)
``` 
 Prediction C:
 ``` 
(raw)(hex){1}(C){TC}_1(comp)[(C){M}<_2](C){PC}_2(r)<_5<&|(raw)(pp)[(C){M}](C){PI}(C){FC}_3(v)<_3(C){FC}_4(v)&<_4|[(C){TC}_5][(C){LC}_6][{tout}(C){PC}_7(v)<_7(prod)]{bout}(v)<_6(hex){2}(C){TC}_8(prod)n|(raw)(C){FC}_9<_5(v)<_9(hex){1}(prod)n|(raw)(hex){2}(v)<_10(prod)
``` 
 Prediction D:
 ``` 
(raw)(hex){1}(C){TC}_1(comp)[(C){M}<_2](C){PC}_2(r)<_5<&|(raw)(pp)[(C){M}](C){PI}(C){FC}_3(v)<_3(C){FC}_4(v)&<_4|[(C){TC}_5][(C){LC}_6][{tout}(C){PC}_7(v)<_7(prod)]{bout}(v)<_6(hex){2}(C){TC}_8(prod)n|(raw)(C){FC}_9<_3(v)<_9(hex){1}(prod)n|(raw)(hex){2}(v)<_10(prod)
```
 Ground Truth PID:
 ``` 
(raw)(hex){1}(C){TC}_1(comp)[(C){M}<_2](C){PC}_2(r)<_5<&|(raw)(pp)[(C){M}](C){PI}(C){FC}_3(v)<_3(C){FC}_4(v)&<_4|[(C){TC}_5][(C){LC}_6][{tout}(C){PC}_7(v)<_7(prod)]{bout}(v)<_6(hex){2}(C){TC}_8(prod)n|(raw)(C){FC}_9<_1(v)<_9(hex){1}(prod)n|(raw)(hex){2}(v)<_8(prod)
``` 
============================================================

 PFD Input 4:
 ``` 
(raw)(v)(hex){1}(rect)<1<4[{tout}(cond)(sep)[(v)(prod)](splt)[(v)(hex){2}(prod)](v)1]{bout}(splt)[(v)(v)(hex){3}(rect)<2<3[{tout}(cond)(sep)[(v)(prod)](splt)[(v)(hex){4}(prod)](v)2]{bout}(splt)[(v)(prod)](hex){5}3](hex){6}4n|(raw)(splt)[(hex){2}(mix)<5(prod)](v)5n|(raw)(splt)[(hex){4}(mix)<6(prod)](v)6n|(raw)(v)(hex){5}(prod)n|(raw)(v)(hex){1}(prod)n|(raw)(v)(hex){6}(prod)n|(raw)(v)(hex){3}(prod)
``` 
 Prediction A:
 ``` 
(raw)(C){FC}_1(v)<_1(hex){1}(C){TC}_2(rect)<1<4[(C){PC}_3][(C){LC}_4][{tout}(cond)(sep)[(C){LC}_5][(v)<_3(prod)](splt)[(v)<_5(hex){2}(C){TC}_6(prod)](C){FC}_7(v)1<_7]{bout}(splt)[(C){FC}_8(v)<_8(C){FC}_9(v)<_9(hex){3}(C){TC}_10(rect)<2<3[(C){PC}_11][(C){LC}_12][{tout}(cond)(sep)[(C){LC}_13][(v)<_11(prod)](splt)[(v)<_13(hex){4}(C){TC}_14(prod)](C){FC}_15(v)2<_15]{bout}(splt)[(v)<_11(prod)](hex){5}3](hex){6}4n|(raw)(splt)[(hex){2}(mix)<5(prod)](v)5<_5n|(raw)(splt)[(hex){4}(mix)<6(prod)](v)6<_1n|(raw)(C){FC}_16(v)<_16(hex){5}(prod)n|(raw)(C){FC}_17(v)<_17(hex){6}(prod)n|(raw)(v)<_8(hex){3}(prod)
``` 
 Prediction B:
 ``` 
(raw)(C){FC}_1(v)<_1(hex){1}(C){TC}_2(rect)<1<4[(C){PC}_3][(C){LC}_4][{tout}(cond)(sep)[(C){LC}_5][(v)<_3(prod)](splt)[(v)<_5(hex){2}(C){TC}_6(prod)](C){FC}_7(v)1<_7]{bout}(splt)[(C){FC}_8(v)<_8(C){FC}_9(v)<_9(hex){3}(C){TC}_10(rect)<2<3[(C){PC}_11][(C){LC}_12][{tout}(cond)(sep)[(C){LC}_13][(v)<_11(prod)](splt)[(v)<_13(hex){4}(C){TC}_14(prod)](C){FC}_15(v)2<_15]{bout}(splt)[(v)<_11(prod)](hex){5}3](hex){6}4n|(raw)(splt)[(hex){2}(mix)<5(prod)](v)5<_5n|(raw)(splt)[(hex){4}(mix)<6(prod)](v)6<_1n|(raw)(C){FC}_16(v)<_16(hex){5}(prod)n|(raw)(C){FC}_17(v)<_17(hex){6}(prod)n|(raw)(v)<_9(hex){3}(prod)
``` 
 Prediction C:
 ``` 
(raw)(C){FC}_1(v)<_1(hex){1}(C){TC}_2(rect)<1<4[(C){PC}_3][(C){LC}_4][{tout}(cond)(sep)[(C){LC}_5][(v)<_3(prod)](splt)[(v)<_5(hex){2}(C){TC}_6(prod)](C){FC}_7(v)1<_7]{bout}(splt)[(C){FC}_8(v)<_8(C){FC}_9(v)<_9(hex){3}(C){TC}_10(rect)<2<3[(C){PC}_11][(C){LC}_12][{tout}(cond)(sep)[(C){LC}_13][(v)<_11(prod)](splt)[(v)<_13(hex){4}(C){TC}_14(prod)](C){FC}_15(v)2<_15]{bout}(splt)[(v)<_11(prod)](hex){5}3](hex){6}4n|(raw)(splt)[(hex){2}(mix)<5(prod)](v)5<_5n|(raw)(splt)[(hex){4}(mix)<6(prod)](v)6<_1n|(raw)(C){FC}_16(v)<_16(hex){5}(prod)n|(raw)(C){FC}_17(v)<_17(hex){6}(prod)n|(raw)(v)<_7(hex){3}(prod)
``` 
 Prediction D:
 ``` 
(raw)(C){FC}_1(v)<_1(hex){1}(C){TC}_2(rect)<1<4[(C){PC}_3][(C){LC}_4][{tout}(cond)(sep)[(C){LC}_5][(v)<_3(prod)](splt)[(v)<_5(hex){2}(C){TC}_6(prod)](C){FC}_7(v)1<_7]{bout}(splt)[(C){FC}_8(v)<_8(C){FC}_9(v)<_9(hex){3}(C){TC}_10(rect)<2<3[(C){PC}_11][(C){LC}_12][{tout}(cond)(sep)[(C){LC}_13][(v)<_11(prod)](splt)[(v)<_13(hex){4}(C){TC}_14(prod)](C){FC}_15(v)2<_15]{bout}(splt)[(v)<_11(prod)](hex){5}3](hex){6}4n|(raw)(splt)[(hex){2}(mix)<5(prod)](v)5<_5n|(raw)(splt)[(hex){4}(mix)<6(prod)](v)6<_1n|(raw)(C){FC}_16(v)<_16(hex){5}(prod)n|(raw)(C){FC}_17(v)<_17(hex){6}(prod)n|(raw)(v)<_15(hex){3}(prod)
``` 

 Prediction E:
 ``` 
(raw)(C){FC}_1(v)<_1(hex){1}(C){TC}_2(rect)<1<4[(C){PC}_3][(C){LC}_4][{tout}(cond)(sep)[(C){LC}_5][(v)<_3(prod)](splt)[(v)<_5(hex){2}(C){TC}_6(prod)](C){FC}_7(v)1<_7]{bout}(splt)[(C){FC}_8(v)<_8(C){FC}_9(v)<_9(hex){3}(C){TC}_10(rect)<2<3[(C){PC}_11][(C){LC}_12][{tout}(cond)(sep)[(C){LC}_13][(v)<_11(prod)](splt)[(v)<_13(hex){4}(C){TC}_14(prod)](C){FC}_15(v)2<_15]{bout}(splt)[(v)<_11(prod)](hex){5}3](hex){6}4n|(raw)(splt)[(hex){2}(mix)<5(prod)](v)5<_1n|(raw)(splt)[(hex){4}(mix)<6(prod)](v)6<_1n|(raw)(C){FC}_16(v)<_16(hex){5}(prod)n|(raw)(C){FC}_17(v)<_17(hex){6}(prod)n|(raw)(v)<_15(hex){3}(prod)
``` 
 Ground Truth PID:
 ``` 
(raw)(C){FC}_1(v)<_1(hex){1}(C){TC}_2(rect)<1<4[(C){PC}_3][(C){LC}_4][(C){TC}_5][{tout}(cond)(sep)[(C){LC}_6][(v)<_3(prod)](splt)[(v)<_6(hex){2}(C){TC}_7(prod)](C){FC}_8(v)1<_8]{bout}(splt)[(C){FC}_9<_5(v)<_9(C){FC}_10(v)<_10(hex){3}(C){TC}_11(rect)<2<3[(C){PC}_12][(C){LC}_13][{tout}(cond)(sep)[(C){LC}_14][(v)<_12(prod)](splt)[(C){FC}_15(v)<_15(hex){4}(C){TC}_16(prod)](v)2<_14]{bout}(splt)[(v)<_13(prod)](hex){5}3](hex){6}4n|(raw)(splt)[(hex){2}(mix)<5(prod)](v)5<_7n|(raw)(splt)[(hex){4}(mix)<6(prod)](v)6<_16n|(raw)(C){FC}_17(v)<_17(hex){5}(prod)n|(raw)(v)<_2(hex){1}(prod)n|(raw)(v)<_4(hex){6}(prod)n|(raw)(v)<_11(hex){3}(prod)
``` 
============================================================

 PFD Input 5:
 ``` 
(raw)(comp)(v)(r)<&|(raw)(hex)&|[{bout}(v)(prod)]{tout}(v)(pp)(v)(prod)
``` 
 Prediction A:
 ``` 
(raw)(comp)[(C){M}<_1](C){PC}_1(C){FC}_2(v)<_2(r)<&|(raw)(comp)[(C){M}<_3](C){PC}_3(C){FC}_4(v)&<_4|[(C){TI}][(C){LC}_5][{bout}(v)<_5(prod)]{tout}(C){PC}_6(v)<_6(pp)[(C){M}](C){PI}(C){FC}_7(v)<_7(prod)
``` 
 Prediction B:
 ``` 
(raw)(hex)<_1(C){TC}_1(r)<_5<&|(raw)(comp)[(C){M}<_2](C){PC}_2(C){FC}_3(v)&<_3|[(C){TI}][(C){LC}_4][{bout}(v)<_4(prod)]{tout}(C){PC}_5(v)<_5(pp)[(C){M}](C){PI}(C){FC}_6(v)<_6(prod)
```

 Prediction C:
 ``` 
(raw)(hex)<_1(C){TC}_1(r)<_5<&|(raw)(comp)[(C){M}<_2](C){PC}_2(C){FC}_3(v)&<_3|[(C){TC}_4][(C){LC}_5][{bout}(v)<_5(prod)]{tout}(C){PC}_6(v)<_6(pp)[(C){M}](C){PI}(C){FC}_7(v)<_7(prod)
``` 
 Prediction D:
 ``` 
(raw)(hex)<_1(C){TC}_1(r)<&|(raw)(comp)[(C){M}<_2](C){PC}_2(C){FC}_3(v)&<_3|[(C){TI}][(C){LC}_4][{bout}(v)<_4(prod)]{tout}(C){PC}_5(v)<_5(pp)[(C){M}](C){PI}(C){FC}_6(v)<_6(prod)
``` 
 Prediction E:
 ``` 
(raw)(hex)<_1(C){TC}_1(r)<_5<&|(raw)(comp)[(C){M}<_2](C){PC}_2(C){FC}_3(v)&<_3|[(C){TI}][(C){LC}_4][{bout}(v)<_4(prod)]{tout}(C){PC}_5(v)<_5(pp)[(C){M}](C){PI}(C){FC}_6(v)<_6(prod)](C){FC}_7(v)<_7(prod)
``` 
 Ground Truth PID:
 ``` 
(raw)(comp)[(C){M}<_1](C){PC}_1(C){FC}_2(v)<_2(r)<&|(raw)(hex)<_3(C){TC}&_3|[(C){TI}][(C){LC}_4][{bout}(v)<_4(prod)]{tout}(C){PC}_5(v)<_5(pp)[(C){M}](C){PI}(C){FC}_6(v)<_6(prod)
``` 



We also tested the output by converting Input 5 and its predictions into a graph and visually comparing it with the ground truth. The generated graph was very similar to the target. With further fine-tuning over more epochs, I strongly believe the model will be able to perfectly replicate the target graph.


INPUT - ![image](https://github.com/user-attachments/assets/fea387b3-46fa-4303-9ae2-6f6df48abf92)

ACTUAL OUTPUT - ![image (1)](https://github.com/user-attachments/assets/cf07887e-6387-42f4-8c92-6ed69b3a43c3)


PREDICTION A - ![image (3)](https://github.com/user-attachments/assets/6ca10c63-8bb8-4264-80c6-e59cb6592f4e)

PREDICTION B - ![image (2)](https://github.com/user-attachments/assets/e827cb81-3619-4bb2-a58a-8e5e62082c01)





# ACCURACY

Calculating the accuracy is a major issue as there can be several correct options and thus comparing the ground truth directly won't be practically correct.

The paper uses Top K accuracy, which calculates the accuracy as how many predictions match the ground truth exactly out of the total predictions. We cannot use this method in our model unless its trained on larger epochs and we start getting exactly correct outputs.

Instead we can use the following metrics to calculate the accuracy:-

We will take the example of input 5 from ByT5 as following :-

 ```
predictions = {
    "A": "(raw)(comp)[(C){M}<_1](C){PC}_1(C){FC}_2(v)<_2(r)<&|(raw)(comp)[(C){M}<_3](C){PC}_3(C){FC}_4(v)&<_4|[(C){TI}][(C){LC}_5][{bout}(v)<_5(prod)]{tout}(C){PC}_6(v)<_6(pp)[(C){M}](C){PI}(C){FC}_7(v)<_7(prod)",
    "B": "(raw)(hex)<_1(C){TC}_1(r)<_5<&|(raw)(comp)[(C){M}<_2](C){PC}_2(C){FC}_3(v)&<_3|[(C){TI}][(C){LC}_4][{bout}(v)<_4(prod)]{tout}(C){PC}_5(v)<_5(pp)[(C){M}](C){PI}(C){FC}_6(v)<_6(prod)",
    "C": "(raw)(hex)<_1(C){TC}_1(r)<_5<&|(raw)(comp)[(C){M}<_2](C){PC}_2(C){FC}_3(v)&<_3|[(C){TC}_4][(C){LC}_5][{bout}(v)<_5(prod)]{tout}(C){PC}_6(v)<_6(pp)[(C){M}](C){PI}(C){FC}_7(v)<_7(prod)",
    "D": "(raw)(hex)<_1(C){TC}_1(r)<&|(raw)(comp)[(C){M}<_2](C){PC}_2(C){FC}_3(v)&<_3|[(C){TI}][(C){LC}_4][{bout}(v)<_4(prod)]{tout}(C){PC}_5(v)<_5(pp)[(C){M}](C){PI}(C){FC}_6(v)<_6(prod)",
    "E": "(raw)(hex)<_1(C){TC}_1(r)<_5<&|(raw)(comp)[(C){M}<_2](C){PC}_2(C){FC}_3(v)&<_3|[(C){TI}][(C){LC}_4][{bout}(v)<_4(prod)]{tout}(C){PC}_5(v)<_5(pp)[(C){M}](C){PI}(C){FC}_6(v)<_6(prod)](C){FC}_7(v)<_7(prod)",
}
ground_truth = "(raw)(comp)[(C){M}<_1](C){PC}_1(C){FC}_2(v)<_2(r)<&|(raw)(hex)<_3(C){TC}&_3|[(C){TI}][(C){LC}_4][{bout}(v)<_4(prod)]{tout}(C){PC}_5(v)<_5(pp)[(C){M}](C){PI}(C){FC}_6(v)<_6(prod)"
```
## 1. Tokenising
   The string is tokenized into logical tokens rather than characters for example, as a whole set like :- '<_5','{bout}', '{LC}', '(prod)'  etc
   The accuracy is then calculated by comparing the tokens in the predicted string to the tokens in the ground truth and the correct matches by the total matches will give us the output

## 2. Exact Match 
   This is comparing the predicted string directly to the ground truth as a whole.

## 3. Character-Level Accuracy
   It uses Python's difflib.SequenceMatcher which finds the longest matching blocks between 2 strings and calculates the accuracy by calculating the ratio of matching characters to total characters.
   ```
   char_accuracy = (number of matching characters) / (avg length of predicted string + ground truth)
   ```
## 4. Levenshtein Distance & Score
   It calculates the character level changes/ edits made in the string (insertion, deletion, substitution ) assigning a score to these and then calculating the acc based on the number of changes made
   ```
dist = Levenshtein.distance(pred, gt)
score = 1 - dist / max(len(pred), len(gt))
```
## 5. BLEU Score - Bilingual Evaluation Understudy 
   It is used to compare overlapping n tokens we will use only n=1 
   ```
BLEU = (number of matching unigrams) / (total unigrams in the predicted string)
```
## Output 

Using Input 5 as an example its accuracy output is as follow:-

```
{'A': {'BLEU Score': 0.7744,
       'Char Accuracy': 0.873,
       'Exact Match': 0,
       'Levenshtein Distance': 35,
       'Levenshtein Score': 0.8259,
       'Token Accuracy': 0.3889},
 'B': {'BLEU Score': 0.9099,
       'Char Accuracy': 0.8179,
       'Exact Match': 0,
       'Levenshtein Distance': 53,
       'Levenshtein Score': 0.7056,
       'Token Accuracy': 0.0833},
 'C': {'BLEU Score': 0.8277,
       'Char Accuracy': 0.7744,
       'Exact Match': 0,
       'Levenshtein Distance': 62,
       'Levenshtein Score': 0.6593,
       'Token Accuracy': 0.0833},
 'D': {'BLEU Score': 0.9325,
       'Char Accuracy': 0.8249,
       'Exact Match': 0,
       'Levenshtein Distance': 51,
       'Levenshtein Score': 0.7119,
       'Token Accuracy': 0.5556},
 'E': {'BLEU Score': 0.8178,
       'Char Accuracy': 0.7704,
       'Exact Match': 0,
       'Levenshtein Distance': 75,
       'Levenshtein Score': 0.6287,
       'Token Accuracy': 0.0833}}
```
   
