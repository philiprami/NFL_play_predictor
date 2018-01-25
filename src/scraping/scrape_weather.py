import requests
from bs4 import BeautifulSoup
import pandas as pd

def return_teams(payload):
	teams = {'vikings':'MIN', 'patriots':'NE', 'redskins':'WAS', 'ravens':'BAL', 'broncos':'DEN', 
			'bills':'BUF', 'steelers':'PIT', 'titans':'TEN', 'falcons':'ATL', 'saints':'NO', 'cheifs':'KC',
			'buccaneers':'TB', 'jets':'NYJ', 'chiefs':'KC', 'panthers':'CAR', 'seahawks':'SEA', 
			'bears':'CHI', 'bengals':'CIN', 'browns':'CLE', 'dolphins':'MIA', 'lions':'DET',
			'raiders':'OAK', 'colts':'IND', 'packers':'GB', '49ers':'SF', 'rams':'LA', 'cardinals':'ARI', 
			'giants':'NYG', 'cowboys':'DAL', 'eagles':'PHI', 'texans':'HOU', 'jaguars':'JAX', 'chargers':'LAC'}

	payload_split = payload.split('-at-')
	home_team = teams[payload_split[0]]
	away_team = teams[payload_split[1]]
	return home_team, away_team, '_'.join(sorted([home_team, away_team]))

year = '2017'
weeks = [str(x) for x in range(1, 18)]
weeks = weeks + ['wildcard-weekend', 'divisional-playoffs', 'conf-championships', 'superbowl']
df = pd.DataFrame()
for week in weeks:
	if week in ['wildcard-weekend', 'divisional-playoffs', 'conf-championships', 'superbowl']:
		url = 'http://nflweather.com/en/week/{}/{}/'.format(year, week)
	else:
		url = 'http://nflweather.com/en/week/{}/week-{}/'.format(year, week)
	response = requests.get(url)
	content = response.content
	soup = BeautifulSoup(content, "html.parser")

	game_links_objects = soup.findAll('td', {'class':'details text-center'})
	for link_object in game_links_objects:
		results = {'Temperature': None,
				   'Wind' : None,
				   'Humidity' : None,
				   'Visibility' : None,
				   'Location' : None,
				   'GameDate' : None,
				   'Team1_Team2' : None,
				   'Home_team' : None,
				   'Away_team' : None,
				   'Stadium' : None,
				   'Surface' : None,
				   'Weather_cat' : None,
				   'IsPlayoffs' : 0
				   }

		link = link_object.encode('utf-8', errors='ignore').split('"')[-2]
		detail_link = 'http://nflweather.com' + link
		detail_response = requests.get(detail_link)
		detail_content = detail_response.content
		detail_soup = BeautifulSoup(detail_content, "html.parser")

		p_tags = detail_soup.findAll('p')
		weather_category = p_tags[3].text.replace('\n', '').strip()
		# stadium = p_tags[14].text
		results['Weather_cat'] = weather_category
		# results['Stadium'] = stadium
		for tag in p_tags:
			tag_text = tag.text
			if 'Temperature' in tag_text:
				temperature = tag_text.split()[1]
				results['Temperature'] = temperature
			if 'Wind' in tag_text:
				wind = tag_text.split()[1]
				results['Wind'] = wind
			if 'Humidity' in tag_text:
				humidity = tag_text.split()[1]
				results['Humidity'] = humidity
			if 'Visibility' in tag_text:
				visibility = tag_text.split()[1]
				results['Visibility'] = visibility
			if 'Location' in tag_text:
				location = tag_text.split(':')[1].strip()
				results['Location'] = location
			if 'Surface' in tag_text:
				surface = tag_text.split()[1]
				results['Surface'] = surface

		description_tag = detail_soup.findAll('div', {'class':'row-fluid centered'})[0]
		datetime = description_tag.findChildren()[1].text.split()[1]
		results['GameDate'] = datetime

		home, away, teams = return_teams(detail_link.split('/')[-1])
		results['Home_team'] = home
		results['Away_team'] = away
		results['Team1_Team2'] = teams
		
		df = df.append(results, ignore_index=True)

df.to_csv('../data/{}_weather.csv'.format(year), index=False)