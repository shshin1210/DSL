% function M = vsmlAvgBird()
function M = vsmlAvgBird()

    % vsmlAvgBird() returns the VSML color cone sensitivity matrix of the 'vs average bird'.
    %
    % The VSML matrix ia formatted [401 x 5]:
    % - 5 columns represent [nm vs sws mws lws]
    % - 401 rows represent the sensitivity values between 300nm and 700nm in 1nm increments.
    %
    % The color cone sensitivity curves are from Endler and Mielke 2005.
    % The double cone sensitivities are not provided;
    % A blank double cone matrix may be concatenated.

    M = [300	0	0	0	0
         301	0	0	0	0
         302	0	0	0	0
         303	0	0	0	0
         304	0	0	0	0
         305	0	0	0	0
         306	0	0	0	0
         307	0	0	0	0
         308	0	0	0	0
         309	0	0	0	0
         310	0	0	0	0
         311	0	0	0	0
         312	0	0	0	0
         313	0	0	0	0
         314	0	0	0	0
         315	0	0	0	0
         316	0.00002333	0	0	0
         317	0.00014856	0	0	0
         318	0.0002716	0	0	0
         319	0.00039242	0	0	0
         320	0.00051098	0	0	0
         321	0.00062724	0	0	0
         322	0.00074121	0	0	0
         323	0.00085288	0	0	0
         324	0.00096228	0	0	0
         325	0.00106943	0	0	0
         326	0.00117438	0	0	0
         327	0.0012772	0	0	0
         328	0.00137797	0	0	0
         329	0.00147678	0	0	0
         330	0.00157373	0	0	0
         331	0.00166897	0	0	0
         332	0.00176263	0	0	0
         333	0.00185487	0	0	0
         334	0.00194585	0	0	0
         335	0.00203576	0	0	0
         336	0.00212478	0	0	0
         337	0.00221312	0	0	0
         338	0.00230099	0	0	0
         339	0.0023886	0	0	0
         340	0.00247618	0	0	0
         341	0.00256396	0	0	0
         342	0.00265215	0	0	0
         343	0.00274099	0	0	0
         344	0.0028307	0	0	0
         345	0.00292151	0	0	0
         346	0.00301363	0	0	0
         347	0.00310728	0	0	0
         348	0.00320265	0	0	0
         349	0.00329994	0	0	0
         350	0.00339933	0	0	0
         351	0.00350098	0	0	0
         352	0.00360505	0	0	0
         353	0.00371168	0	0	0
         354	0.00382098	0	0	0
         355	0.00393306	0	0	0
         356	0.004048	0	0	0
         357	0.00416586	0	0	0
         358	0.00428668	0	0	0
         359	0.0044105	0	0	0
         360	0.00453732	0	0	0
         361	0.00466712	0	0	0
         362	0.00479986	0	0	0
         363	0.00493549	0	0	0
         364	0.00507393	0	0	0
         365	0.0052151	0	0	0
         366	0.00535888	0	0	0
         367	0.00550514	0	0	0
         368	0.00565375	0	0	0
         369	0.00580454	0	0	0
         370	0.00595735	0	0	0
         371	0.00611198	0	0	0
         372	0.00626826	0	0	0
         373	0.00642596	0	0	0
         374	0.00658488	0	0	0
         375	0.0067448	0	0	0
         376	0.00690548	0	0	0
         377	0.0070667	0	0	0
         378	0.00722822	0	0	0
         379	0.0073898	0	0	0
         380	0.00755118	0	0	0
         381	0.00771214	0	0	0
         382	0.00787242	0	0	0
         383	0.00803177	0	0	0
         384	0.00818996	0	0	0
         385	0.00834673	0	0	0
         386	0.00850185	0	0	0
         387	0.00865507	0	0	0
         388	0.00880616	0	0	0
         389	0.00895487	0	0	0
         390	0.00910099	0	0	0
         391	0.00924426	0	0	0
         392	0.00938447	0	0	0
         393	0.00952139	0	0	0
         394	0.00965479	0	0	0
         395	0.00978445	0	0	0
         396	0.00991016	0	0	0
         397	0.01003168	0	0	0
         398	0.01014881	0	0	0
         399	0.01026133	0	0	0
         400	0.01036903	0	0	0
         401	0.01047168	0	0	0
         402	0.01056909	0	0	0
         403	0.01066103	0	0	0
         404	0.0107473	0	0	0
         405	0.01082768	0	0	0
         406	0.01090196	0	0	0
         407	0.01096993	0	0	0
         408	0.0110314	0	0	0
         409	0.01108614	0	0	0
         410	0.01113395	0	0	0
         411	0.01117462	0	0	0
         412	0.01120797	0	0	0
         413	0.01123378	0	0	0
         414	0.01125187	0	0	0
         415	0.01126205	0	0	0
         416	0.01126413	0	0	0
         417	0.01125794	0	0	0
         418	0.01124331	0	0	0
         419	0.01122009	0	0	0
         420	0.01118812	0	0	0
         421	0.01114728	0	0	0
         422	0.01109745	0	0	0
         423	0.01103851	0	0	0
         424	0.01097038	0	0	0
         425	0.010893	0	0	0
         426	0.01080632	0	0	0
         427	0.01071031	0	0	0
         428	0.01060497	0.00000001	0	0
         429	0.01049031	0.00000002	0	0
         430	0.0103664	0.00000007	0	0
         431	0.01023331	0.0000002	0	0
         432	0.01009114	0.00000055	0	0
         433	0.00994003	0.00000137	0	0
         434	0.00978015	0.00000317	0	0
         435	0.0096117	0.00000684	0	0
         436	0.00943489	0.00001386	0	0
         437	0.00925001	0.00002644	0	0
         438	0.00905735	0.0000478	0	0
         439	0.00885722	0.00008223	0	0
         440	0.00865001	0.00013515	0	0
         441	0.0084361	0.00021307	0	0
         442	0.00821592	0.0003233	0	0
         443	0.00798993	0.00047366	0	0
         444	0.00775863	0.00067199	0	0
         445	0.00752252	0.00092566	0	0
         446	0.00728217	0.00124107	0	0
         447	0.00703815	0.00162315	0	0
         448	0.00679107	0.00207507	0	0
         449	0.00654154	0.00259793	0	0
         450	0.00629023	0.00319072	0	0
         451	0.00603779	0.00385031	0	0
         452	0.00578491	0.00457163	0	0
         453	0.00553228	0.00534791	0	0
         454	0.00528061	0.006171	0	0
         455	0.0050306	0.00703175	0	0
         456	0.00478294	0.00792031	0	0
         457	0.00453834	0.00882655	0	0
         458	0.00429747	0.00974034	0	0
         459	0.00406099	0.01065185	0	0
         460	0.00382953	0.01155176	0	0
         461	0.00360369	0.01243144	0	0
         462	0.00338402	0.0132831	0	0
         463	0.00317104	0.01409984	0	0
         464	0.00296519	0.0148757	0	0
         465	0.0027669	0.0156057	0	0
         466	0.00257648	0.01628575	0	0
         467	0.00239423	0.01691267	0	0
         468	0.00222035	0.01748412	0	0
         469	0.002055	0.01799849	0	0
         470	0.00189825	0.0184549	0	0
         471	0.00175011	0.01885304	0	0
         472	0.00161055	0.01919317	0	0
         473	0.00147946	0.01947601	0	0
         474	0.0013567	0.01970266	0	0
         475	0.00124205	0.01987457	0	0
         476	0.00113528	0.01999345	0	0
         477	0.00103612	0.02006126	0	0
         478	0.00094426	0.02008009	0	0
         479	0.00085936	0.02005222	0	0
         480	0.0007811	0.01997997	0	0
         481	0.0007091	0.01986578	0	0
         482	0.00064301	0.0197121	0	0
         483	0.00058246	0.01952141	0	0
         484	0.0005271	0.01929619	0	0
         485	0.00047658	0.01903889	0	0
         486	0.00043053	0.01875195	0	0
         487	0.00038864	0.01843775	0	0
         488	0.00035059	0.01809865	0	0
         489	0.00031606	0.01773691	0.00000001	0
         490	0.00028477	0.01735478	0.00000003	0
         491	0.00025645	0.01695439	0.00000009	0
         492	0.00023084	0.01653784	0.00000023	0
         493	0.00020771	0.01610716	0.00000056	0
         494	0.00018684	0.01566426	0.00000124	0
         495	0.00016801	0.01521104	0.0000026	0
         496	0.00015104	0.01474928	0.00000519	0
         497	0.00013576	0.01428071	0.00000984	0
         498	0.00012201	0.01380697	0.00001781	0
         499	0.00010963	0.01332964	0.00003088	0
         500	0.00009851	0.01285022	0.00005146	0
         501	0.00008851	0.01237016	0.00008263	0
         502	0.00007952	0.0118908	0.00012818	0
         503	0.00007144	0.01141345	0.00019259	0
         504	0.00006419	0.01093934	0.00028089	0
         505	0.00005768	0.01046963	0.00039852	0
         506	0.00005183	0.0100054	0.00055111	0
         507	0.00004658	0.0095477	0.00074418	0
         508	0.00004187	0.00909747	0.00098289	0
         509	0.00003764	0.00865563	0.00127175	0
         510	0.00003384	0.00822298	0.00161431	0
         511	0.00003043	0.0078003	0.00201305	0
         512	0.00002737	0.00738827	0.00246909	0
         513	0.00002463	0.00698753	0.00298223	0
         514	0.00002216	0.00659863	0.00355081	0
         515	0.00001994	0.00622205	0.00417186	0
         516	0.00001796	0.00585821	0.00484112	0
         517	0.00001617	0.00550745	0.00555323	0
         518	0.00001456	0.00517005	0.00630192	0
         519	0.00001312	0.00484622	0.00708018	0
         520	0.00001183	0.00453608	0.00788054	0
         521	0.00001066	0.00423971	0.0086952	0
         522	0.00000961	0.00395711	0.00951631	0
         523	0.00000867	0.00368822	0.01033611	0
         524	0.00000782	0.00343292	0.01114711	0
         525	0.00000706	0.00319103	0.01194222	0
         526	0.00000637	0.00296232	0.01271488	0
         527	0.00000576	0.00274652	0.01345911	0
         528	0.0000052	0.00254331	0.01416957	0
         529	0.0000047	0.00235233	0.01484163	0
         530	0.00000425	0.00217318	0.01547136	0
         531	0.00000384	0.00200545	0.0160555	0
         532	0.00000348	0.00184869	0.01659151	0
         533	0.00000314	0.00170245	0.01707745	0
         534	0.00000285	0.00156625	0.01751203	0
         535	0.00000258	0.0014396	0.01789449	0
         536	0.00000233	0.00132203	0.01822462	0
         537	0.00000211	0.00121305	0.01850263	0
         538	0.00000192	0.00111217	0.01872918	0
         539	0.00000174	0.00101893	0.01890526	0
         540	0.00000158	0.00093286	0.0190322	0
         541	0.00000143	0.0008535	0.01911157	0.00000001
         542	0.0000013	0.00078043	0.01914515	0.00000002
         543	0.00000118	0.0007132	0.01913493	0.00000004
         544	0.00000107	0.00065144	0.01908302	0.00000008
         545	0.00000097	0.00059473	0.01899162	0.00000017
         546	0.00000088	0.00054273	0.01886303	0.00000033
         547	0.0000008	0.00049508	0.01869958	0.00000063
         548	0.00000073	0.00045145	0.01850362	0.00000117
         549	0.00000066	0.00041153	0.01827753	0.00000209
         550	0.0000006	0.00037504	0.01802363	0.0000036
         551	0.00000055	0.00034169	0.01774424	0.00000603
         552	0.0000005	0.00031124	0.01744162	0.00000981
         553	0.00000046	0.00028345	0.01711798	0.00001552
         554	0.00000042	0.0002581	0.01677546	0.00002394
         555	0.00000038	0.00023498	0.01641613	0.00003601
         556	0.00000035	0.00021392	0.016042	0.00005295
         557	0.00000032	0.00019472	0.01565496	0.00007617
         558	0.00000029	0.00017724	0.01525686	0.00010734
         559	0.00000026	0.00016132	0.01484944	0.00014834
         560	0.00000024	0.00014682	0.01443436	0.00020127
         561	0.00000022	0.00013363	0.0140132	0.00026838
         562	0.0000002	0.00012163	0.01358743	0.00035204
         563	0.00000018	0.00011071	0.01315848	0.00045467
         564	0.00000017	0.00010078	0.01272765	0.00057866
         565	0.00000015	0.00009174	0.01229621	0.00072633
         566	0.00000014	0.00008352	0.01186531	0.00089982
         567	0.00000013	0.00007605	0.01143605	0.00110103
         568	0.00000012	0.00006925	0.01100944	0.00133154
         569	0.00000011	0.00006307	0.01058644	0.00159258
         570	0.0000001	0.00005745	0.01016792	0.00188493
         571	0.00000009	0.00005233	0.0097547	0.00220896
         572	0.00000008	0.00004768	0.00934753	0.00256453
         573	0.00000008	0.00004345	0.0089471	0.00295103
         574	0.00000007	0.0000396	0.00855405	0.00336737
         575	0.00000006	0.0000361	0.00816894	0.00381201
         576	0.00000006	0.00003292	0.0077923	0.00428298
         577	0.00000005	0.00003002	0.0074246	0.00477792
         578	0.00000005	0.00002738	0.00706625	0.00529413
         579	0.00000004	0.00002498	0.0067176	0.00582864
         580	0.00000004	0.00002279	0.00637898	0.00637823
         581	0.00000004	0.0000208	0.00605065	0.00693953
         582	0.00000003	0.00001899	0.00573283	0.00750905
         583	0.00000003	0.00001734	0.00542569	0.00808324
         584	0.00000003	0.00001584	0.00512937	0.00865856
         585	0.00000003	0.00001447	0.00484394	0.00923149
         586	0.00000002	0.00001322	0.00456947	0.00979864
         587	0.00000002	0.00001209	0.00430594	0.01035673
         588	0.00000002	0.00001105	0.00405333	0.01090263
         589	0.00000002	0.0000101	0.00381158	0.01143344
         590	0.00000002	0.00000924	0.00358057	0.01194645
         591	0.00000002	0.00000845	0.00336019	0.01243919
         592	0.00000002	0.00000774	0.00315025	0.01290943
         593	0.00000001	0.00000708	0.00295058	0.0133552
         594	0.00000001	0.00000648	0.00276095	0.01377477
         595	0.00000001	0.00000594	0.00258112	0.0141667
         596	0.00000001	0.00000544	0.00241085	0.01452977
         597	0.00000001	0.00000498	0.00224984	0.014863
         598	0.00000001	0.00000457	0.0020978	0.01516568
         599	0.00000001	0.00000419	0.00195444	0.01543729
         600	0.00000001	0.00000384	0.00181943	0.01567754
         601	0.00000001	0.00000352	0.00169246	0.01588634
         602	0.00000001	0.00000323	0.00157319	0.01606378
         603	0.00000001	0.00000297	0.0014613	0.01621011
         604	0.00000001	0.00000272	0.00135645	0.01632576
         605	0.00000001	0.0000025	0.00125831	0.01641128
         606	0	0.0000023	0.00116656	0.01646734
         607	0	0.00000211	0.00108086	0.01649475
         608	0	0.00000194	0.00100091	0.01649438
         609	0	0.00000178	0.00092638	0.0164672
         610	0	0.00000164	0.00085698	0.01641426
         611	0	0.00000151	0.00079241	0.01633666
         612	0	0.00000138	0.00073239	0.01623553
         613	0	0.00000127	0.00067663	0.01611206
         614	0	0.00000117	0.00062488	0.01596746
         615	0	0.00000108	0.00057689	0.01580295
         616	0	0.00000099	0.0005324	0.01561977
         617	0	0.00000092	0.0004912	0.01541915
         618	0	0.00000084	0.00045307	0.01520232
         619	0	0.00000078	0.00041778	0.0149705
         620	0	0.00000072	0.00038516	0.01472491
         621	0	0.00000066	0.00035501	0.01446671
         622	0	0.00000061	0.00032716	0.01419707
         623	0	0.00000056	0.00030145	0.01391713
         624	0	0.00000052	0.00027771	0.01362797
         625	0	0.00000048	0.00025581	0.01333068
         626	0	0.00000044	0.00023562	0.01302628
         627	0	0.00000041	0.000217	0.01271576
         628	0	0.00000038	0.00019983	0.01240009
         629	0	0.00000035	0.00018402	0.01208019
         630	0	0.00000032	0.00016945	0.01175694
         631	0	0.0000003	0.00015603	0.01143117
         632	0	0.00000027	0.00014367	0.0111037
         633	0	0.00000025	0.00013229	0.01077529
         634	0	0.00000023	0.00012181	0.01044666
         635	0	0.00000022	0.00011216	0.0101185
         636	0	0.0000002	0.00010329	0.00979146
         637	0	0.00000019	0.00009512	0.00946617
         638	0	0.00000017	0.0000876	0.00914318
         639	0	0.00000016	0.00008068	0.00882305
         640	0	0.00000015	0.00007431	0.0085063
         641	0	0.00000014	0.00006845	0.00819338
         642	0	0.00000013	0.00006306	0.00788475
         643	0	0.00000012	0.0000581	0.00758081
         644	0	0.00000011	0.00005354	0.00728194
         645	0	0.0000001	0.00004934	0.0069885
         646	0	0.00000009	0.00004547	0.0067008
         647	0	0.00000009	0.00004192	0.00641913
         648	0	0.00000008	0.00003864	0.00614375
         649	0	0.00000007	0.00003563	0.00587489
         650	0	0.00000007	0.00003286	0.00561276
         651	0	0.00000006	0.00003031	0.00535754
         652	0	0.00000006	0.00002796	0.00510938
         653	0	0.00000006	0.00002579	0.00486841
         654	0	0.00000005	0.0000238	0.00463472
         655	0	0.00000005	0.00002196	0.00440841
         656	0	0.00000004	0.00002027	0.00418951
         657	0	0.00000004	0.00001872	0.00397807
         658	0	0.00000004	0.00001728	0.0037741
         659	0	0.00000004	0.00001596	0.00357757
         660	0	0.00000003	0.00001474	0.00338847
         661	0	0.00000003	0.00001362	0.00320673
         662	0	0.00000003	0.00001259	0.0030323
         663	0	0.00000003	0.00001163	0.00286508
         664	0	0.00000003	0.00001075	0.00270498
         665	0	0.00000002	0.00000994	0.00255186
         666	0	0.00000002	0.00000919	0.00240562
         667	0	0.00000002	0.0000085	0.00226609
         668	0	0.00000002	0.00000787	0.00213312
         669	0	0.00000002	0.00000728	0.00200656
         670	0	0.00000002	0.00000674	0.00188623
         671	0	0.00000002	0.00000623	0.00177193
         672	0	0.00000001	0.00000577	0.0016635
         673	0	0.00000001	0.00000534	0.00156073
         674	0	0.00000001	0.00000495	0.00146343
         675	0	0.00000001	0.00000459	0.00137141
         676	0	0.00000001	0.00000425	0.00128444
         677	0	0.00000001	0.00000394	0.00120235
         678	0	0.00000001	0.00000365	0.00112492
         679	0	0.00000001	0.00000338	0.00105196
         680	0	0.00000001	0.00000314	0.00098326
         681	0	0.00000001	0.00000291	0.00091863
         682	0	0.00000001	0.0000027	0.00085788
         683	0	0.00000001	0.00000251	0.00080082
         684	0	0.00000001	0.00000232	0.00074726
         685	0	0.00000001	0.00000216	0.00069703
         686	0	0.00000001	0.000002	0.00064994
         687	0	0.00000001	0.00000186	0.00060584
         688	0	0	0.00000173	0.00056456
         689	0	0	0.00000161	0.00052594
         690	0	0	0.00000149	0.00048982
         691	0	0	0.00000139	0.00045608
         692	0	0	0.00000129	0.00042455
         693	0	0	0.0000012	0.00039513
         694	0	0	0.00000111	0.00036766
         695	0	0	0.00000104	0.00034205
         696	0	0	0.00000096	0.00031817
         697	0	0	0.0000009	0.0002959
         698	0	0	0.00000084	0.00027516
         699	0	0	0.00000078	0.00025585
         700	0	0	0.00000072	0.00023786];
end