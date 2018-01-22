import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver

# seasons = [str(i) for i in range(2012, 2018)]
seasons = ['2017']
weeks = [str(i) for i in range(16, 21)]
columns = ['quarter','gameTime','down','to_go','description', 'away_team', 
		   'week', 'home_score','away_score','yard_line', 'home_team', 
		   'gameDate', 'seasonYear', 'offenseTeam', 'away_team_wins',
		   'home_team_wins', 'stadium'
		   ]

df = pd.DataFrame(columns=columns)

for season in seasons:
	for week in weeks:
		url = 'https://www.pro-football-reference.com/years/{}/week_{}.htm'.format(season, week)
		response = requests.get(url)
		content = response.content
		soup = BeautifulSoup(content, "html.parser")
		game_summaries_soup = soup.find('div', {'class':'game_summaries'})
		link_tags = game_summaries_soup.findAll('td', {'class': 'right gamelink'})

		for game_tag in link_tags:
			# each link_tag represents a game, find game links contains in tags
			# iterate through all game pages, extract game information
			game_df = pd.DataFrame(columns=columns)

			link_object = game_tag.findChildren()[0]
			link_object_str = link_object.encode('utf-8')
			link = link_object_str.split('"')[1]

			# need to use a driver to access inforamtion contained in dynamic tables
			driver = webdriver.Chrome('/Users/Philip/Desktop/chromedriver')
			game_url = 'https://www.pro-football-reference.com' + link
			driver.get(game_url)
			src = driver.page_source
			parser = BeautifulSoup(src,"lxml")

			# start dict to contain values
			game_info = {'home_team': None,
						 'away_team' : None,
						 'gameDate' : None,
						 'seasonYear' : season,
						 'week' : week,
						 'away_team_wins' : None,
						 'home_team_wins' : None,
						 'stadium' : None,
						 'home_qb' : None,
						 'away_qb' : None,
						 'home_rb' : None,
						 'away_rb' : None
						 }

			# extract home and away team names from header
			# also extract gameDate
			info_tag = parser.find('h1').text
			info_tag_split = info_tag.split('-')
			if len(info_tag_split) > 2:
				playoff_round, teams, dateTime = info_tag_split
			else:
				teams, dateTime = info_tag_split
			away_team, home_team = teams.split(' at ')
			game_info['home_team'] = home_team.strip()
			game_info['away_team'] = away_team.strip()
			game_info['gameDate'] = dateTime.strip()

			# from the scorebox, scrape total wins for each team as well as 
			# the stadium that the game was played in
			score_box_tag = parser.find('div', {'class' : 'scorebox'})
			home_score_tag, away_score_tag = score_box_tag.findChildren('div', {'class' : "score"})
			home_team_wins = home_score_tag.findNext().text.split('-')[0]
			away_team_wins = away_score_tag.findNext().text.split('-')[0]
			game_info['home_team_wins'] = home_team_wins
			game_info['away_team_wins'] = away_team_wins
			strong_tags = score_box_tag.findChildren('strong')
			for s_tag in strong_tags:
				if s_tag.text == 'Stadium':
					game_info['stadium'] = s_tag.findNext().text

			# create a new dataframe to store the drive starttimes
			# this will be used later to determine which team is on offense for each play
			drives_df = pd.DataFrame()

			home_drives_tag = parser.find('table', {'id' : "home_drives"})
			home_rows = home_drives_tag.findAll('tr')[1:]
			for drive in home_rows:
				attributes = {e.attrs['data-stat'] : e.text for e in drive}
				results = {'quarter' : attributes['quarter'],
						   'time_start' : attributes['time_start'],
						   'offenseTeam' : game_info['home_team']}

				drives_df = drives_df.append(results, ignore_index=True)

			away_drives_tag = parser.find('table', {'id' : "vis_drives"})
			away_rows = away_drives_tag.findAll('tr')[1:]
			for drive in away_rows:
				attributes = {e.attrs['data-stat'] : e.text for e in drive}
				results = {'quarter' : attributes['quarter'],
						   'time_start' : attributes['time_start'],
						   'offenseTeam' : game_info['away_team']}
						   
				drives_df = drives_df.append(results, ignore_index=True)

			# retrive starting quarterback and running back for each team
			starters_tag_home = parser.find('div', {'id' : "all_home_starters"})
			starters_table_home = starters_tag_home.findChild('div', {'id' : 'div_home_starters'})
			qb_tag_home, rb_tag_home = starters_table_home.findChildren('tr')[1:3]
			qb_home = qb_tag_home.findNext().text
			rb_home = rb_tag_home.findNext().text
			game_info['home_qb'] = qb_home
			game_info['home_rb'] = rb_home

			starters_tag_away = parser.find('div', {'id' : "all_vis_starters"})
			starters_table_away = starters_tag_away.findChild('div', {'id' : 'div_vis_starters'})
			qb_tag_away, rb_tag_away = starters_table_away.findChildren('tr')[1:3]
			qb_away = qb_tag_away.findNext().text
			rb_away = rb_tag_away.findNext().text
			game_info['away_qb'] = qb_away
			game_info['away_rb'] = rb_away

			# iterate through the entire play by play, append to gamedict
			list_of_attributes = {"id" : "pbp"}
			play_table = parser.findAll('table', attrs=list_of_attributes)[0]
			plays = play_table.findAll('tr')
			plays = plays[2:-1]

			for play in plays:
				play_dict = {'quarter': None,
							 'gameTime' : None,
							 'down' : None,
							 'to_go' : None,
							 'description' : None,
							 'home_score' : None,
							 'away_score' : None,
							 'yard_line' : None,
							 'offenseTeam' : None
							 }

				fields = play.findChildren()
				attributes = {e.attrs['data-stat'] : e.text for e in fields if e.has_attr('data-stat') and e['data-stat'] != 'onecell'}
				if len(attributes) == 0:
					continue
				try:
					int(attributes['pbp_score_hm'])
				except Exception:
					continue

				play_dict['quarter'] = attributes['quarter']
				play_dict['gameTime'] = attributes['qtr_time_remain']
				play_dict['down'] = attributes['down']
				play_dict['to_go'] = attributes['yds_to_go']
				play_dict['description'] = attributes['detail']
				play_dict['home_score'] = attributes['pbp_score_hm']
				play_dict['away_score'] = attributes['pbp_score_aw']
				play_dict['yard_line'] = attributes['location']

				play_dict.update(game_info)
				game_df = game_df.append(play_dict, ignore_index=True)

			# finally, merge game_df with drives_df
			# this will give us each offensive team for each play
			# Replace csv with each new successful game scraped
			drives_df['time_start'] = drives_df['time_start'].apply(lambda x: x[1:] if x[0] == '0' and len(x) == 5 else x)
			OT_mask = game_df['quarter'] == 'OT'
			game_df.loc[OT_mask, 'quarter'] = u'5'
			merged_df = pd.merge(game_df, drives_df, how='left', left_on=['quarter', 'gameTime'], right_on=['quarter', 'time_start'])
			merged_df = merged_df.ffill()
			merged_df = merged_df.drop(['time_start', 'offenseTeam_x'], axis=1)
			merged_df = merged_df.rename(index=str, columns={'offenseTeam_y' : 'offenseTeam'})
			df = df.append(merged_df)
			df.to_csv('../data/scraped/scraped_{}.csv'.format(season), index=False)
			driver.close()

