import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Total_Months_Spend_in_2017':62.252, 'Total_Offnet_Spend':3563, 'Total_Sms_Spend':6.76,'TotalOnnetspend':0,'TotalUniqueCalls':13,'is_Uxaa_network':0,'Total_Data_Consumption':19.1533,'Total_Data_spend':11.25})

print(r.json())