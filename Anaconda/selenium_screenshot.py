from io import StringIO
from io import BytesIO
from PIL import Image
#from pyvirtualdisplay import Display
from selenium import webdriver
#from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

#display = Display(visible=0, size=(1600, 27000))
#display.start()

browser = webdriver.Firefox(capabilities=firefox_capabilities)

browser.maximize_window()
browser.get('http://rilyaus.com')


"""url = ''
browser.implicitly_wait(10)
browser.get(url)"""

#selector = ''
#element = browser.find_element_by_css_selector(selector)
xpath = '//div[@class="team-info"]/div[@class="info-box"]'
element = browser.find_element_by_xpath(xpath)

location, size = element.location_once_scrolled_into_view, element.size
left, top = location['x'], location['y']
width, height = size['width'], size['height']
box = (int(left), int(top), int(left + width), int(top + height))

screenshot_path = 'screenshot' + '.png'

screenshot = browser.get_screenshot_as_base64()
image = Image.open(BytesIO(base64.b64decode(screenshot)))
output = image.crop(box)
output.save(screenshot_path, 'PNG')

browser.quit()
display.stop()
