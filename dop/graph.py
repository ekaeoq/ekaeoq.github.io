import matplotlib.pyplot as plt

#cba wiht this shit rucno bum izracunal i del vure priblizno jesus fuck
pon=33+30+15+26+34+19+40+5015:27
26:19
34:27
19:24
40:28
50:06
uto=2
sri=2
cet=4
pet=2
sub=3
ned=1
kolko = [1,2,3,4,5,6,7]
vioko= [pon,uto,sri,cet,pet,sub,ned]
tick_label = ['Monday',' Tuesday',' Wednesday',' Thursday',' Friday',' Saturday',' Sunday']
plt.bar(kolko, vioko, tick_label = tick_label, width = 0.4, color = ['blue'])

plt.ylabel('Minutes wasted')
plt.xlabel('Wasted on each day')
plt.title('What a week should not look like')
plt.show()
