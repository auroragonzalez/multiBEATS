
TRAIN_FILES = ['../data/ArabicDigits/', 	 			 	   # 0
	       '../data/ArabicDigits_SL3-WD3-nfeat2-red0.333/', 		   # 1
	       '../data/AUSLAN/',                                                  # 2
	       '../data/AUSLAN_SL4-WD5-nfeat2-red0.316/',                          # 3
	       '../data/CharacterTrajectories/',                                   # 4
	       '../data/CharacterTrajectories_SL4-WD15-nfeat3-red0.33/',           # 5
	       '../data/CMUsubject16/',                                            # 6
	       '../data/CMUsubject16_SL5-WD10-nfeat3-red0.351/',                   # 7
	       '../data/ECG/',                                                     # 8
	       '../data/ECG_SL3-WD8-nfeat2-red0.332/',                             # 9
	       '../data/JapaneseVowels/',                                          # 10
	       '../data/JapaneseVowels_SL3-WD7-nfeat2-red0.385/',                  # 11
	       '../data/Libras/',                                                  # 12
	       '../data/Libras_SL2-WD16-nfeat2-red0.333/',                         # 13
	       '../data/Wafer/',                                                   # 14
	       '../data/Wafer_SL3-WD3-nfeat2-red0.321/',                           # 15
	       '../data/ArabicDigits_SL4-WD3-nfeat2-red0.504/',			   # 16
	       '../data/AUSLAN_SL3-WD25-nfeat2-red0.212/',			#17
	       '../data/CharacterTrajectories_SL3-WD35-nfeat3-red0.166/',       #18
	       '../data/CMUsubject16_SL4-WD35-nfeat3-red0.234/',		#19
	       '../data/ECG_SL2-WD30-nfeat2-red0.156/',				#20
	       '../data/JapaneseVowels_SL3-WD3-nfeat2-red0.293/',		#21
	       '../data/Libras_SL2-WD10-nfeat2-red0.203/',			#22
	       '../data/Wafer_SL2-WD40-nfeat2-red0.213/',			#23
	       '../data/ArabicDigits_SL5-WD4-nfeat2-red0.611/',			#24
	       '../data/AUSLAN_SL5-WD6-nfeat2-red0.440/',			#25
	       '../data/CharacterTrajectories_SL5-WD12-nfeat3-red0.429/',       #26
	       '../data/CMUsubject16_SL6-WD10-nfeat3-red0.468/',		#27
	       '../data/ECG_SL4-WD5-nfeat2-red0.496/',				#28
	       '../data/JapaneseVowels_SL4-WD5-nfeat2-red0.461/',		#29
	       '../data/Libras_SL3-WD10-nfeat2-red0.467/',			#30
	       '../data/Wafer_SL4-WD6-nfeat2-red0.506/'				#31
                ]


TEST_FILES = ['../data/ArabicDigits/',                                            # 0
	      '../data/ArabicDigits_SL3-WD3-nfeat2-red0.333/',                    # 1
	      '../data/AUSLAN/',                                                  # 2
	      '../data/AUSLAN_SL4-WD5-nfeat2-red0.316/',                           # 3
	      '../data/CharacterTrajectories/',                                   # 4
	      '../data/CharacterTrajectories_SL4-WD15-nfeat3-red0.33/',          # 5
	      '../data/CMUsubject16/',                                            # 6
	      '../data/CMUsubject16_SL5-WD10-nfeat3-red0.351/',                   # 7
	      '../data/ECG/',                                                     # 8
	      '../data/ECG_SL3-WD8-nfeat2-red0.332/',                            # 9
	      '../data/JapaneseVowels/',                                          # 10
	      '../data/JapaneseVowels_SL3-WD7-nfeat2-red0.385/',                    # 11
	      '../data/Libras/',                                                  # 12
	      '../data/Libras_SL2-WD16-nfeat2-red0.333/',                         # 13
	      '../data/Wafer/',                                                   # 14
	      '../data/Wafer_SL3-WD3-nfeat2-red0.321/'     ,                       # 15
	      '../data/ArabicDigits_SL4-WD3-nfeat2-red0.504/',			  #16
	       '../data/AUSLAN_SL3-WD25-nfeat2-red0.212/',			#17
	       '../data/CharacterTrajectories_SL3-WD35-nfeat3-red0.166/',       #18
	       '../data/CMUsubject16_SL4-WD35-nfeat3-red0.234/',		#19
	       '../data/ECG_SL2-WD30-nfeat2-red0.156/',				#20
	       '../data/JapaneseVowels_SL3-WD3-nfeat2-red0.293/',		#21
	       '../data/Libras_SL2-WD10-nfeat2-red0.203/',			#22
	       '../data/Wafer_SL2-WD40-nfeat2-red0.213/',			#23
	       '../data/ArabicDigits_SL5-WD4-nfeat2-red0.611/',			#24
	       '../data/AUSLAN_SL5-WD6-nfeat2-red0.440/',			#25
	       '../data/CharacterTrajectories_SL5-WD12-nfeat3-red0.429/',       #26
	       '../data/CMUsubject16_SL6-WD10-nfeat3-red0.468/',		#27
	       '../data/ECG_SL4-WD5-nfeat2-red0.496/',				#28
	       '../data/JapaneseVowels_SL4-WD5-nfeat2-red0.461/',		#29
	       '../data/Libras_SL3-WD10-nfeat2-red0.467/',			#30
	       '../data/Wafer_SL4-WD6-nfeat2-red0.506/'				#31
		      ]


MAX_NB_VARIABLES = [13,  # 0
                    13,  # 1
                    22,  # 2
                    22,  # 3
                    3,   # 4
                    3,   # 5
                    62,  # 6
                    62,  # 7
                    2,   # 8
                    2,   # 9
                    12,  # 10
                    12,  # 11
                    2,   # 12
                    2,   # 13
                    6,   # 14
                    6,   # 15
                    13,  #16
                    22,  #17
                    3,  #18
                    62, #19
                    2,  #20
                    12, #21
                    2,  #22
                    6,  #23
                    13, #24
                    22, #25
                    3,  #26
                    62, #27
                    2,  #28
                    12, #29
                    2,  #30
                    6,  #31
                    ]



MAX_TIMESTEPS_LIST = [93,   # 0
                     62,   # 1
                     96,   # 2
                     66,   # 3
                     205,  # 4
                     144,  # 5
                     534,  # 6
                     345,  # 7
                     147,  # 8
                     98,  # 9
                     26,   # 10
                     16,   # 11
                     45,   # 12
                     30,   # 13
                     198,  # 14
                     132,  # 15
                     46,   #16
                     76,   #17
                     171,   #18
                     411,   #19
                     124,   #20
                     18,   #21
                     36,   #22
                     160,   #23
                     36,   #24
                     54,  #25
                     117, #26
                     288, #27
                     74,  #28
                     14,  #29
                     24,  #30
                     98,  #31
                     ]



NB_CLASSES_LIST = [10,  # 0
                   10,  # 1
                   95,  # 2
                   95,  # 3
                   20,  # 4
                   20,  # 5
                   2,   # 6
                   2,   # 7
                   2,   # 8
                   2,   # 9
                   9,   # 10
                   9,   # 11
                   15,  # 12
                   15,  # 13
                   2,   # 14
                   2,   # 15
                   10,  # 16
                   95,  # 17
                   20,  # 18
                   2,   #19 
                   2,   #20
                   9,   #21
                   15,  #22
                   2,   #23
                   10,  #24
                   95,  #25
                   20,  #26
                   2,   #27 
                   2,   #28
                   9,   #29
                   15,  #30
                   2,   #31
                   ]

