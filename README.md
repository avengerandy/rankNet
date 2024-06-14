
```log
$ python src/train.py 
trainTestWithSameRankingPage
epoch0 trainCurrentRate: 0.65
epoch1 trainCurrentRate: 0.69
epoch2 trainCurrentRate: 0.70
testCurrentRate: 0.71
=============================
trainTestWithOtherRankingPage (original)
epoch0 trainCurrentRate: 0.64
epoch1 trainCurrentRate: 0.69
epoch2 trainCurrentRate: 0.71
testCurrentRate: 0.59
=============================
trainTestWithOtherRankingPage (modified)
epoch0 trainCurrentRate: 0.56
epoch1 trainCurrentRate: 0.61
epoch2 trainCurrentRate: 0.63
testCurrentRate: 0.63
```

```log
$ python src/hypothesisTesting.py 
sameRankingPage:
KruskalResult(statistic=1.0812276101266591, pvalue=0.2984230935024443)
KruskalResult(statistic=1.2293546877250272, pvalue=0.2675325726507133)
KruskalResult(statistic=4.739247376673478, pvalue=0.02948196620438234)
KruskalResult(statistic=0.2498109742924989, pvalue=0.6172082077070977)
KruskalResult(statistic=0.1342936115544768, pvalue=0.7140211629799282)
KruskalResult(statistic=3.7993428755934544, pvalue=0.051272701398055905)
KruskalResult(statistic=2.3208088635951003, pvalue=0.12765363132515045)
KruskalResult(statistic=0.20121346233637416, pvalue=0.6537431519333335)
KruskalResult(statistic=0.16173364059725473, pvalue=0.6875653860322029)
KruskalResult(statistic=0.00029763515339625575, pvalue=0.9862354939706918)
KruskalResult(statistic=2.6357363673068495, pvalue=0.10448360817354285)
KruskalResult(statistic=5.334489670983912, pvalue=0.020907460507742472)
================================================================
otherRankingPage:
KruskalResult(statistic=5.707629175795012, pvalue=0.016891336683534482)
KruskalResult(statistic=5.662375604642771, pvalue=0.017332631063601354)
KruskalResult(statistic=112.85008127691606, pvalue=2.3272265345661586e-26)
KruskalResult(statistic=0.1183531380745446, pvalue=0.7308275534196667)
KruskalResult(statistic=25.143537615271654, pvalue=5.321769076621548e-07)
KruskalResult(statistic=17.96143852177434, pvalue=2.2542565418223413e-05)
KruskalResult(statistic=10.723489544548972, pvalue=0.0010578398572342951)
KruskalResult(statistic=7.095480025151965, pvalue=0.007727859157872331)
KruskalResult(statistic=1.2862035283540991, pvalue=0.25674877258145284)
KruskalResult(statistic=5.224849009453471, pvalue=0.022266379457297883)
KruskalResult(statistic=0.9354796779285227, pvalue=0.3334430705576221)
KruskalResult(statistic=54.25046497908814, pvalue=1.7649537781872029e-13)
```

```log
$ python src/experiment.py
originalCurrentRateRecords
originalMean: 0.59
originalStd: 0.01
==========================
modifiedCurrentRateRecords
modifiedMean: 0.62
modifiedStd: 0.01
==========================
hypothesisTesting
Ttest_indResult(statistic=-7.422601959112843, pvalue=6.640167006148289e-09)
KruskalResult(statistic=22.41274160255207, pvalue=2.199102600927154e-06)
```

```log
trainTestWithSameRankingPage
$ python src/eval.py
0.9401113752916035

trainTestWithOtherRankingPage (original)
$ python src/eval.py
0.9051408339908514

trainTestWithOtherRankingPage (modified)
$ python src/eval.py
0.9353777722502167
```
