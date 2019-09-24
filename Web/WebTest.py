from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
#driver = webdriver.Chrome()
driver = webdriver.Firefox()

#driver.get("http://www.python.org")
driver.get("https://time.knightec.se/cgi-bin/Maconomy/MaconomyPortal.macoprod.S_KNI_MCS.exe/Framework/maconomyportalmain.msc")

#window_before = driver.window_handles[0]
#window_after = driver.window_handles[1]

#print(driver.title)
#https://time.knightec.se/cgi-bin/Maconomy/MaconomyPortal.macoprod.S_KNI_MCS.exe/Framework/maconomyportalmain.msc?javaVersion=25.51-b03

#https://time.knightec.se/cgi-bin/Maconomy/MaconomyPortal.macoprod.S_KNI_MCS.exe/Framework/maconomyportalmain.msc
usern = "mawah"
passw = "FernandoMacon10"

delay = 10

wait = WebDriverWait(driver, 10)

time.sleep(5)

#driver.find_element_by_xpath('/html/body/form/table/tbody/tr[1]/td[2]/input')


#print(window_before)


#driver.switch_to_window(window_before)


#WebDriverWait(driver, delay).until(EC.presence_of_element_located(('xpath', '/html/body/form/table/tbody/tr[1]/td[2]/input')))

driver.switch_to.frame(driver.find_element_by_css_selector("frame[name='mainwindow']"))

#WebDriverWait(driver, delay).until(EC.presence_of_element_located(('name', 'mainwindow')))

WebDriverWait(driver, delay).until(EC.presence_of_element_located(('name', 'nameofuser')))

username = driver.find_element_by_name('nameofuser')
password = driver.find_element_by_name("password")

username.send_keys(usern)
password.send_keys(passw)
password.send_keys(Keys.ENTER)

time.sleep(10)

wait = WebDriverWait(driver, 10)

print(driver.window_handles)

driver.switch_to_default_content()

driver.switch_to.frame(driver.find_element_by_css_selector("frame[name='mainwindow']"))

driver.switch_to.frame(driver.find_element_by_css_selector("frame[name='portalmain']"))

driver.switch_to.frame(driver.find_element_by_xpath("/html/frameset/frame"))

#driver.switch_to.frame(driver.find_element_by_css_selector("frame[name='S399a_rightside']"))
#MenuAndRightside > frame:nth-child(2)

driver.switch_to.frame(driver.find_element_by_css_selector("frame[name='componentframe']"))

WebDriverWait(driver, delay).until(EC.presence_of_element_located(('xpath', '//*[@id="b:timeSheetCard:createTimeSheet"]')))

# Create log
button = driver.find_element_by_xpath('//*[@id="b:timeSheetCard:createTimeSheet"]')
button.click()

buttonNew = driver.find_element_by_xpath('//*[@id="timeSheetTable"]/tbody/tr/th[1]/img')
buttonNew.click()

time.sleep(5)
wait = WebDriverWait(driver, 5)

mon = '8'
monBox = driver.find_element_by_xpath('//*[@id="timeSheetTable"]/tbody/tr[2]/td[7]/input')
monBox.send_keys(mon)

tue = '8'
tueBox = driver.find_element_by_xpath('//*[@id="timeSheetTable"]/tbody/tr[2]/td[8]/input')
tueBox.send_keys(tue)

wed = '8'
wedBox = driver.find_element_by_xpath('//*[@id="timeSheetTable"]/tbody/tr[2]/td[9]/input')
wedBox.send_keys(wed)


thu = '8'
thuBox = driver.find_element_by_xpath('//*[@id="timeSheetTable"]/tbody/tr[2]/td[10]/input')
thuBox.send_keys(thu)

fri = '8'
friBox = driver.find_element_by_xpath('//*[@id="timeSheetTable"]/tbody/tr[2]/td[11]/input')
friBox.send_keys(fri)

assignmentname = '415'
assignment = driver.find_element_by_xpath('//*[@id="timeSheetTable:2"]') #
assignment.send_keys(assignmentname)

projectname = '1024870-10'
projectbox = driver.find_element_by_xpath('//*[@id="timeSheetTable:1"]')
projectbox.send_keys(projectname)

#assignment = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="timeSheetTable:2"]')))

#select = Select(driver.find_element_by_xpath('/html/body/table/tbody/tr[4]/td/table/tbody/tr[2]/td/table/tbody/tr/td/table/tbody/tr[2]/td/table/tbody/tr[1]/td/table/tbody/tr[2]/td[4]/table/tbody/tr/td[2]/img'))
#select.select_by_visible_text(assignmentname)
# select by value
#select.select_by_value('2')

time.sleep(60)
driver.close()
