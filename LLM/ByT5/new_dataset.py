import pandas as pd
df1=pd.read_json("Training Data Public Upload/eval_data_1k_wo_valves.json", lines=True)
df2=pd.read_json("Training Data Public Upload/eval_data_1k.json", lines=True)

print(df1.columns.tolist())

df1=df1.drop(columns=['PID'])
df2=df2.drop(columns=['PID'])

df2=df2.rename(columns={'PFD':'PID'})

combined_df = pd.DataFrame({
    'PFD': df1['PFD'].reset_index(drop=True),
    'PID': df2['PID'].reset_index(drop=True)
})
combined_df.to_json('eval_data_1k_wv2v.json', orient='records', indent=2)




