import re

text1 = '2017-01	Realizado	D.DR.PB.GD0101010103	SENAI1.3.2.13	DR.PB.SENAI.31020101	D1.1.2.2.1	A.PC.001	-0,007106603782501'
text2 = '2017-01	Realizado	D.DR.PB.GD0101010103	SENAI1.3.2.17	DR.PB.SENAI.31020201	D1.1.2.2.2	A.ADM.011	-7,81886693943148'
text3 = '2017-01	Realizado	D.DR.PB.GD0101010102	SENAI1.5	DR.PB.SENAI.31020701	D1.1.3.1.3	A.GP.007	-0,000336820753708'

if re.search(r'SENAI1', text1):
    print('match')
else:
    print('doesn´t match')

if re.search(r'SENAI1\.\d\.\d\.\d', text2):
    print('match')
else:
    print('doesn´t match')

if re.search(r'SENAI1\.\d\.\d\.\d', text3):
    print('match')
else:
    print('doesn´t match')
