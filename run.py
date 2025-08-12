from ApiClean import run 
overall_score = run()


title = overall_score['ticker']
score = overall_score['score']



print(f"For ticker {title}, the overall sentiment score is {score}")


