# NFL_play_predictor

My project is related to the National Football League (NFL) and predicting plays. Football has plenty of formations, strategies, and gameplay variations based on specific game circumstances, available players (and their respective strengths), and coaching tendencies. I want to build a model to predict the type of play that will be called based on game situation and formation. For example, if the team on offense is down by 10 points with 4 minutes to play and they’re in a shotgun formation, will they pass or run? 

Data
1.	Data source: http://nflsavant.com/ . There are csv files from the last five years with play by play breakdowns of every game. I plan to supplement the data by scraping for features like weather, coach, offensive coordinators, playoff picture, and any other relevant factors I can think of.
2.	Extract features: Team, opponent, coach, formation, play style indicators, personnel, game situation, number of timeouts, field position, quarter, score …
3.	Extract target: Play type. To start I’ll try to predict run or pass. Eventually I’d like to be able to predict subcategories like short pass, deep pass, screen, fake, run left, run right…
