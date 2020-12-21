from flask import Flask,render_template,request
import pandas as pd
import jieba
import jieba.analyse
import jieba.posseg as pseg
import re
from openpyxl import Workbook
import numpy as np
from numpy import *
import xlwt


app=Flask(__name__)


@app.route('/',methods=['POST','GET'])   #這一行是告訴 ‘/’允許的method有什麼

def index():
	if request.method == 'POST':  # 告訴系統要請求post的資料
		if request.values['send'] == '送出':  # 按下送出的話，才執行下面的東西

			sentment_table = pd.read_excel('VAD-Lexicon.xlsx')  # 匯入情緒辭典
			# sentment_table.drop(['Unnamed: 10','Unnamed: 11'],inplace=True,axis=1)
			all_table = pd.read_excel('VAD-Lexicon.xlsx', sheet_name='sheetALL')  # 定義工作表

			val_dict = dict(zip(list(all_table.word), list(all_table.Valence)))  # 讀取Valence
			aro_dict = dict(zip(list(all_table.word), list(all_table.Arousal)))  # 讀取Arousal
			dom_dict = dict(zip(list(all_table.word), list(all_table.Dominance)))  # 讀取Dominance
			euc_dict = dict(zip(list(all_table.word), list(all_table.euc)))  # 讀取歐式距離
			emo_dict = dict(zip(list(all_table.word), list(all_table.emotion)))  # 讀取情緒

			sentment_dict = {**val_dict}
			sentment_dict1 = {**aro_dict}
			sentment_dict2 = {**dom_dict}
			sentment_dicte = {**euc_dict}
			sentment_dict_emo = {**emo_dict}

			def is_number(s):
				try:
					float(s)
					return True
				except ValueError:
					pass

				try:
					import unicodedata
					unicodedata.numeric(s)
					return True
				except (TypeError, ValueError):
					pass

				return False

			# 藉由自定義函數is_number()方法來判斷字符串是否為數字

			for v in sentment_dict.keys():
				if is_number(v):
					pass
				else:
					jieba.suggest_freq(v, True)

			for a in sentment_dict1.keys():
				if is_number(a):
					pass
				else:
					jieba.suggest_freq(a, True)

			for d in sentment_dict2.keys():
				if is_number(d):
					pass
				else:
					jieba.suggest_freq(d, True)
			# 把不是數字的字典的單字加進結巴分詞

			for e in sentment_dicte.keys():
				if is_number(e):
					pass
				else:
					jieba.suggest_freq(e, True)

			for emo in sentment_dict_emo.keys():
				if is_number(emo):
					pass
				else:
					jieba.suggest_freq(emo, True)

			stop_words = [re.findall(r'\S+', v)[0] for v in open('stopwords_ch.txt', encoding='utf8').readlines() if
						  len(re.findall(r'\S+', v)) > 0]

			def Vsent2word(sentence, stop_words=stop_words):
				words = jieba.cut(sentence, HMM=False)
				words = [v for v in words if v not in stop_words]
				return words

			stop_words = [re.findall(r'\S+', a)[0] for a in open('stopwords_ch.txt', encoding='utf8').readlines() if
						  len(re.findall(r'\S+', a)) > 0]

			def Asent2word(sentence, stop_words=stop_words):
				words = jieba.cut(sentence, HMM=False)
				words = [a for a in words if a not in stop_words]
				return words

			stop_words = [re.findall(r'\S+', d)[0] for d in open('stopwords_ch.txt', encoding='utf8').readlines() if
						  len(re.findall(r'\S+', d)) > 0]

			def Dsent2word(sentence, stop_words=stop_words):
				words = jieba.cut(sentence, HMM=False)
				words = [d for d in words if d not in stop_words]
				return words

			# 準備斷詞和停止詞

			def get_sentment(userText):
				global tokens
				global valence_score
				global arousal_score
				global dominance_score
				global emotion
				tokens = Vsent2word(userText)  # 斷詞
				Vscore = 0  # V情緒分數
				countword = 0  # 詞塊數量
				all_word = 0
				non_emo_word = 0

				for v in tokens:  # 從tokens(斷詞)中取得元素執行v，直到元素取完為止
					if v in sentment_dict.keys():
						# print(v, "的V分數為", sentment_dict[v])
						for emo in tokens:
							if emo in sentment_dict_emo.keys():
								if emo == v:
									# print(emo,"的情緒是",sentment_dict_emo[emo])
									all_word += 1
									if sentment_dict_emo[emo] == "neutral":
										non_emo_word += 1
				if non_emo_word != 0:
					print(userText, "的非情緒詞的數量為", non_emo_word)
					print("非情緒詞在本句的佔比為", non_emo_word / all_word)
				else:
					return 0

				# print(tokens)
				# print(type(tokens))
				for v in tokens:  # 從tokens(斷詞)中取得元素執行v，直到元素取完為止
					if v in sentment_dict.keys():
						print(v, "的V分數為", sentment_dict[v])
						for e in tokens:
							if e in sentment_dicte.keys():
								for emo in tokens:
									if emo in sentment_dict_emo.keys():
										if emo == v:
											if non_emo_word / all_word < 0.25:  # 當非情緒詞佔小於75%的時候
												if sentment_dict_emo[emo] == "neutral":
													sentment_dict[v] = 0
													if e == v:
														# print(e, "的歐式距離為", sentment_dicte[e])
														if 0 <= sentment_dicte[e] < 0.1:
															t = 3
														elif 0.1 <= sentment_dicte[e] < 0.2:
															t = 2
														elif 0.2 <= sentment_dicte[e] < 0.5:
															t = 1
														elif 0.5 <= sentment_dicte[e] <= 0.9:
															t = 0.25
														else:
															t = 0
														print(e, "的情緒為", sentment_dict_emo[emo], "，與該情緒的歐式距離為",
															  sentment_dicte[e], "，權重為", t)
														sentment_dict[v] *= t
														print(v, "加權後的V分數為", sentment_dict[v])
														print("\n")

														Vscore += sentment_dict[v]  # 隨著情緒分數的斷詞增加，將V分數疊加上去
														countword += t
												else:
													if e == v:
														# print(e,"的歐式距離為", sentment_dicte[e])
														if 0 <= sentment_dicte[e] < 0.1:
															t = 3
														elif 0.1 <= sentment_dicte[e] < 0.2:
															t = 2
														elif 0.2 <= sentment_dicte[e] < 0.5:
															t = 1
														elif 0.5 <= sentment_dicte[e] <= 1:
															t = 0.25
														else:
															t = 0
														print(e, "的情緒為", sentment_dict_emo[emo], "，與該情緒的歐式距離為",
															  sentment_dicte[e], "，權重為", t)
														sentment_dict[v] *= t
														print(v, "加權後的V分數為", sentment_dict[v])
														print("\n")

														Vscore += sentment_dict[v]  # 隨著情緒分數的斷詞增加，將V分數疊加上去
														countword += t
											else:
												if e == v:
													if 0 <= sentment_dicte[e] < 0.1:
														t = 3
													elif 0.1 <= sentment_dicte[e] < 0.2:
														t = 2
													elif 0.2 <= sentment_dicte[e] < 0.5:
														t = 1
													elif 0.5 <= sentment_dicte[e] <= 1:
														t = 0.25
													else:
														t = 0

													print(e, "的情緒為", sentment_dict_emo[emo])
													print(e, "與該情緒的歐式距離為", sentment_dicte[e], "，權重為", t)
													sentment_dict[v] *= t
													print(v, "加權後的V分數為", sentment_dict[v])
													Vscore += sentment_dict[v]  # 隨著情緒分數的斷詞增加，將V分數疊加上去
													countword += t
				if countword != 0:
					print(Vscore / countword)  # 將疊加完的V分數依斷詞個數相除
					print("_______")
					print("\n")
				else:
					print("\n")
					return 0
				# -----------------------------------------------------

				tokens = Asent2word(userText)  # 斷詞
				Ascore = 0  # A情緒分數
				countword1 = 0  # 詞塊數量
				all_word1 = 0
				non_emo_word1 = 0

				for a in tokens:  # 從tokens(斷詞)中取得元素執行a，直到元素取完為止
					if a in sentment_dict1.keys():
						# print(a, "的A分數為", sentment_dict1[a])
						for emo in tokens:
							if emo in sentment_dict_emo.keys():
								if emo == a:
									# print(emo,"的情緒是",sentment_dict_emo[emo])
									all_word1 += 1
									if sentment_dict_emo[emo] == "neutral":
										non_emo_word1 += 1
				if non_emo_word1 != 0:
					print(userText, "的非情緒詞的數量為", non_emo_word1)
					print("非情緒詞在本句的佔比為", non_emo_word1 / all_word1)
				else:
					return 0

				for a in tokens:  # 從tokens(斷詞)中取得元素執行a，直到元素取完為止
					if a in sentment_dict1.keys():
						print(a, "的A分數為", sentment_dict1[a])
						for e in tokens:
							if e in sentment_dicte.keys():
								for emo in tokens:
									if emo in sentment_dict_emo.keys():
										if emo == a:
											if non_emo_word1 / all_word1 < 0.25:  # 當非情緒詞佔小於75%的時候
												if sentment_dict_emo[emo] == "neutral":
													sentment_dict1[a] = 0
													if e == a:
														if 0 <= sentment_dicte[e] < 0.1:
															t = 3
														elif 0.1 <= sentment_dicte[e] < 0.2:
															t = 2
														elif 0.2 <= sentment_dicte[e] < 0.5:
															t = 1
														elif 0.5 <= sentment_dicte[e] <= 0.9:
															t = 0.25
														else:
															t = 0

														print(e, "的情緒為", sentment_dict_emo[emo], "，與該情緒的歐式距離為",
															  sentment_dicte[e], "，權重為", t)
														sentment_dict1[a] *= t
														print(a, "加權後的A分數為", sentment_dict1[a])
														print("\n")
														Ascore += sentment_dict1[a]  # 隨著情緒分數的斷詞增加，將A分數疊加上去
														countword1 += t
												else:
													if e == a:
														# print(e,"的歐式距離為", sentment_dicte[e])
														if 0 <= sentment_dicte[e] < 0.1:
															t = 3
														elif 0.1 <= sentment_dicte[e] < 0.2:
															t = 2
														elif 0.2 <= sentment_dicte[e] < 0.5:
															t = 1
														elif 0.5 <= sentment_dicte[e] <= 1:
															t = 0.25
														else:
															t = 0
														print(e, "的情緒為", sentment_dict_emo[emo], "，與該情緒的歐式距離為",
															  sentment_dicte[e], "，權重為", t)
														sentment_dict1[a] *= t
														print(a, "加權後的A分數為", sentment_dict1[a])
														print("\n")
														Ascore += sentment_dict1[a]  # 隨著情緒分數的斷詞增加，將A分數疊加上去
														countword1 += t
											else:
												if e == a:
													if 0 <= sentment_dicte[e] < 0.1:
														t = 3
													elif 0.1 <= sentment_dicte[e] < 0.2:
														t = 2
													elif 0.2 <= sentment_dicte[e] < 0.5:
														t = 1
													elif 0.5 <= sentment_dicte[e] <= 1:
														t = 0.25
													else:
														t = 0

													print(e, "的情緒為", sentment_dict_emo[emo], "，與該情緒的歐式距離為",
														  sentment_dicte[e], "，權重為", t)
													sentment_dict1[a] *= t
													print(a, "加權後的A分數為", sentment_dict1[a])
													print("\n")
													Ascore += sentment_dict1[a]  # 隨著情緒分數的斷詞增加，將A分數疊加上去
													countword1 += t
				if countword1 != 0:
					print(Ascore / countword1)  # 將疊加完的A分數依斷詞個數相除
					print("_______")
					print("\n")
				else:
					print("\n")
					return 0
				# ____________________________________________________________________________

				tokens = Dsent2word(userText)  # 斷詞
				Dscore = 0  # D情緒分數
				countword2 = 0  # 詞塊數量
				all_word2 = 0
				non_emo_word2 = 0

				for d in tokens:  # 從tokens(斷詞)中取得元素執行d，直到元素取完為止
					if d in sentment_dict2.keys():
						# print(d, "的D分數為", sentment_dict2[d])
						for emo in tokens:
							if emo in sentment_dict_emo.keys():
								if emo == d:
									# print(emo,"的情緒是",sentment_dict_emo[emo])
									all_word2 += 1
									if sentment_dict_emo[emo] == "neutral":
										non_emo_word2 += 1
				if non_emo_word2 != 0:
					print(userText, "的非情緒詞的數量為", non_emo_word2)
					print("非情緒詞在本句的佔比為", non_emo_word2 / all_word2)
				else:
					return 0

				for d in tokens:  # 從tokens(斷詞)中取得元素執行d，直到元素取完為止
					if d in sentment_dict2.keys():
						print(d, "的D分數為", sentment_dict2[d])
						for e in tokens:
							if e in sentment_dicte.keys():
								for emo in tokens:
									if emo in sentment_dict_emo.keys():
										if emo == d == e:
											if non_emo_word2 / all_word2 < 0.25:
												if sentment_dict_emo[emo] == "neutral":
													sentment_dict2[d] = 0
													if e == d:
														if 0 <= sentment_dicte[e] < 0.1:
															t = 3
														elif 0.1 <= sentment_dicte[e] < 0.2:
															t = 2
														elif 0.2 <= sentment_dicte[e] < 0.5:
															t = 1
														elif 0.5 <= sentment_dicte[e] <= 0.9:
															t = 0.25
														else:
															t = 0

														print(e, "的情緒為", sentment_dict_emo[emo], "，與該情緒的歐式距離為",
															  sentment_dicte[e], "，權重為", t)
														sentment_dict2[d] *= t
														print(d, "加權後的D分數為", sentment_dict2[d])
														print("\n")
														Dscore += sentment_dict2[d]  # 隨著情緒分數的斷詞增加，將A分數疊加上去
														countword2 += t
												else:
													if e == d:
														# print(e,"的歐式距離為", sentment_dicte[e])
														if 0 <= sentment_dicte[e] < 0.1:
															t = 3
														elif 0.1 <= sentment_dicte[e] < 0.2:
															t = 2
														elif 0.2 <= sentment_dicte[e] < 0.5:
															t = 1
														elif 0.5 <= sentment_dicte[e] <= 1:
															t = 0.25
														else:
															t = 0
														print(e, "的情緒為", sentment_dict_emo[emo], "，與該情緒的歐式距離為",
															  sentment_dicte[e], "，權重為", t)
														sentment_dict2[d] *= t
														print(d, "加權後的D分數為", sentment_dict2[d])
														print("\n")

														Dscore += sentment_dict2[d]  # 隨著情緒分數的斷詞增加，將V分數疊加上去
														countword2 += t
											else:
												if e == d:
													if 0 <= sentment_dicte[e] < 0.1:
														t = 3
													elif 0.1 <= sentment_dicte[e] < 0.2:
														t = 2
													elif 0.2 <= sentment_dicte[e] < 0.5:
														t = 1
													elif 0.5 <= sentment_dicte[e] <= 1:
														t = 0.25
													else:
														t = 0

													print(e, "的情緒為", sentment_dict_emo[emo], "，與該情緒的歐式距離為",
														  sentment_dicte[e], "，權重為", t)
													sentment_dict2[d] *= t
													print(d, "加權後的D分數為", sentment_dict2[d])
													print("\n")

													Dscore += sentment_dict2[d]  # 隨著情緒分數的斷詞增加，將V分數疊加上去
													countword2 += t

				if countword2 != 0:
					print(Dscore / countword2)  # 將疊加完的D分數依斷詞個數相除
					print("_______")
					# -----------------------------------------------------------------------------------------------------------------------------
					non_count = userText.count("沒有") + userText.count("不是") + userText.count(
						"無") + userText.count("大可不必") + userText.count("犯不著") + userText.count(
						"不可以") + userText.count("不可") + userText.count("不能") + userText.count(
						"不再") + userText.count("不得") + userText.count("不行") + userText.count(
						"不准") + userText.count("不許") + userText.count("不必") + userText.count(
						"不用") + userText.count("不須") + userText.count("絕不") + userText.count(
						"決不") + userText.count("犯不著") + userText.count("不可以") + userText.count(
						"不可") + userText.count("不能") + userText.count("不再") + userText.count(
						"不得") + userText.count("不行") + userText.count("不准") + userText.count(
						"不許") + userText.count("不必") + userText.count("不用") + userText.count(
						"不須") + userText.count("絕不") + userText.count("決不") + userText.count(
						"並非") + userText.count("從不") + userText.count("從未") + userText.count(
						"毫不") + userText.count("毫無") + userText.count("絕非") + userText.count("無法")


					if (non_count % 2) == 1:
						print("否定詞有", non_count, "個，是奇數，該句為否定句")
						valence_score = 1 - (Vscore / countword)
						arousal_score = 1 - (Ascore / countword1)
						dominance_score = 1 - (Dscore / countword2)
						Gscore = (valence_score, arousal_score, dominance_score)
						print("該句為否定句，VAD數值為：", Gscore)

						Gscore = (valence_score, arousal_score, dominance_score)
						if 0 <= valence_score <= 0.3 and 0.62 <= arousal_score <= 0.92 and 0.19 <= dominance_score <= 0.49:
							emotion = 'fear'
						elif 0.04 <= valence_score <= 0.34 and 0.6 <= arousal_score <= 0.9 and 0.37 <= dominance_score <= 0.67:
							emotion = 'angry'
						elif 0 <= valence_score <= 0.3 and 0.15 <= arousal_score <= 0.45 and 0.187 <= dominance_score <= 0.487:
							emotion = 'disgust'
						elif 0 <= valence_score <= 0.3 and 0.4 <= arousal_score <= 0.7 and 0.1 <= dominance_score <= 0.4:
							emotion = 'sad'
						elif 0.7 <= valence_score <= 1 and 0.59 <= arousal_score <= 0.89 and 0.57 <= dominance_score <= 0.87:
							emotion = 'happy'
						elif 0.67 <= valence_score <= 0.97 and 0.67 <= arousal_score <= 0.97 and 0.47 <= dominance_score <= 0.77:
							emotion = 'surprise'
						else:
							emotion = 'neutral'
					# print("該否定句的情緒判斷為：",emotion)

					else:
						print("否定詞有", non_count, "個，是偶數，故該句為肯定句或雙重否定")
						valence_score = Vscore / countword
						arousal_score = Ascore / countword1
						dominance_score = Dscore / countword2
						score = (valence_score, arousal_score, dominance_score)
						print("該句為肯定句，VAD數值為：", score)
						score = (valence_score, arousal_score, dominance_score)
						if 0 <= valence_score <= 0.3 and 0.62 <= arousal_score <= 0.92 and 0.19 <= dominance_score <= 0.49:
							emotion = 'fear'
							yv = 0.15
							ya = 0.77
							yd = 0.34
						elif 0.04 <= valence_score <= 0.34 and 0.6 <= arousal_score <= 1 and 0.37 <= dominance_score <= 0.67:
							emotion = 'angry'
							yv = 0.19
							ya = 0.8
							yd = 0.52
						elif 0 <= valence_score <= 0.3 and 0.28 <= arousal_score <= 0.58 and 0.187 <= dominance_score <= 0.487:
							emotion = 'disgust'
							yv = 0.15
							ya = 0.43
							yd = 0.338
						elif 0.05 <= valence_score <= 0.35 and 0.3 <= arousal_score <= 0.7 and 0.1 <= dominance_score <= 0.4:
							emotion = 'sad'
							yv = 0.2
							ya = 0.55
							yd = 0.25
						elif 0.7 <= valence_score <= 1 and 0 <= arousal_score <= 1 and 0.57 <= dominance_score <= 0.87:
							emotion = 'happy'
							yv = 0.85
							ya = 0.74
							yd = 0.72
						elif 0.67 <= valence_score <= 0.97 and 0.67 <= arousal_score <= 0.97 and 0.47 <= dominance_score <= 0.77:
							emotion = 'surprise'
							yv = 0.82
							ya = 0.82
							yd = 0.62
						else:
							emotion = 'neutral'
							yv = 0.5
							ya = 0.5
							yd = 0.5
						# print("該肯定句的情緒判斷為：",emotion)
						print(emotion)


				else:
					return 0

			userText = request.values.get('user')
			print(userText)
			get_sentment(userText)
			val = str(valence_score)
			aro = str(arousal_score)
			dom = str(dominance_score)

			headings = (("斷詞","V","A","D","Emotion"))

			data = ((tokens,val,aro,dom,emotion))

			print(data)
			# print("hk4g4hk4gk4ghk4gk4",valence_score)
			# print(valence_score)
			# print(arousal_score)
			# print(dominance_score)
			# print(emotion)
			return render_template('index.html',headings=headings,data=data,name=request.values['user'])
	return render_template('index.html', name="")

if __name__ == '__main__':
	app.run(host='localhost',port='5000', threaded=True, debug=True)