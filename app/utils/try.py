import requests, re
import numpy as np
u='https://api.stackexchange.com/2.2/similar'
tag_arr=[]
def clean_text(text):
    text = str(text)
    text = re.sub(r"[^\w]", " ", text.lower())
    return text
# for i in range(1,28):
for i in range(1,2):
    print(i)
    p={'page':str(i), 'pagesize':'100','fromdate':'1388534400','tagged':'javascript;node.js;npm','title':'node how to run node app js', 'order':'desc','sort':'votes','min':'40','site':'stackoverflow','key':'hWdB8OaWM0hGZP3sRV18iA(('}
    r = requests.get(url = u, params = p)
    data = r.json()
    temp = data['items']
    for k in temp:
        # tag_arr.append(k['title'].encode('latin1').decode('utf-8'))
        tag_arr.append(k['title'])
        
print(tag_arr)

#np.savetxt('ques-so.txt', tag_arr, delimiter=',', newline='\n', fmt='%s')

with open('ques-tmp.txt', 'w') as f:
    for item in tag_arr:
        f.write("%s\n" % clean_text( item))
    f.close()
