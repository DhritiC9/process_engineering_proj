1. I tried to test the T5-small model and how it works by implementing a T5 base code and also a code from github about tagging using the T5 model. The codes can be found in the files T5_basecode_test.py and T5small_Tag_Finetune_test.ipynb respectively
2. T5_PFD-PID.ipynb = Next I wrote a basic code to solve the problem of converting PFD SFILES TO PID SFILES based on the datasetprovided in the paper. Pls note that i have used the same datsets in all the codes as following:- train = train_data_10k.json, test=test_data_1k.json, eval- eval_data_1k.json. The model used was T5- small and the parameters used were  output_dir="./t5_pid_model",
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    report_to="none",  # Change to "tensorboard" if you want logs
    fp16=torch.cuda.is_available(),
   and my output was as following - === Inference test ===
Input PFD: (raw)(hex){1}(hex){2}(mix)<2(r)[{tout}(v)(prod)]{bout}(v)(splt)[(hex){2}(hex){3}(pp)(v)(mix)<1(r)[{bout}(v)(prod)]{tout}(v)(splt)[(hex){4}(r)[{tout}(v)(prod)]{bout}(v)(hex){4}(prod)](v)1](v)2n|(raw)(hex){1}(v)(prod)n|(raw)(hex){3}(v)(prod)
Predicted PID: (raw)(hex)1(C)TC_1(hex)2(C)TC_2(mix)2(r)_3[(C)TC_3][(C)LC_4][tout(C)PC_5(v)_5(prod)]bout(v)_5(splt)[(hex)2(hex)3(C)TC_7(pp)[(C)M](C)PI(C)FC_8(v)_8(mix)1(r)_9[(C)TC_9][(C)LC_10][bout(v)_10(prod)]tout(C)PC_11(v)_11(splt)[(hex)2(C)TC_12(pp)[(C)M
True PID: (raw)(hex){1}(C){TC}_1(hex){2}(mix)<2(r)<_2[(C){TC}_2][(C){LC}_3][{tout}(C){PC}_4(v)<_4(prod)]{bout}(v)<_3(splt)[(hex){2}(hex){3}(C){TC}_5(pp)[(C){M}](C){PI}(C){FC}_6(v)<_6(mix)<1(r)<_7[(C){TC}_7][(C){LC}_8][{bout}(v)<_8(prod)]{tout}(C){PC}_9(v)<_9(splt)[(C){FC}_10(v)1<_10](hex){4}(r)<_11[(C){TC}_11][(C){LC}_12][{tout}(C){PC}_13(v)<_13(prod)]{bout}(v)<_12(hex){4}(prod)](C){FC}_14(v)2<_14n|(raw)(hex){1}(v)<_1(prod)n|(raw)(hex){3}(v)<_5(prod)
Out here we can visually notice in the output that all the brackets are 
