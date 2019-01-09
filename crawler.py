import requests
from bs4 import BeautifulSoup as Bf
import re


def find_single_line_in_bodys(soup):
    bodys = soup.find_all("tei:body")
    print(bodys)
    for body in bodys:
        for child in body.children:
            print(child.string)


class CrawlerError(ValueError):
    pass


cites_regex = re.compile("cite nums ((?:(?:\d+),)+\d+)")
url = "http://www.perseus.tufts.edu/hopper/CTS?request=GetPassage&urn=urn:cts:greekLit:tlg0012.tlg002.perseus-grc1:{}.{}-"

book = 1
cite_number = 1
while True:
    r = requests.get(url.format(book, cite_number))

    if r.status_code == 200:
        # print(r.text)
        soup = Bf(r.text, "html.parser")
        # print(soup)
        message = soup.find("cts:message")
        if message is not None:
            # print(message.text)
            warning = list(message)[0]
            # print(warning)
            match = re.findall(cites_regex, warning)
            if match is not None:
                try:
                    cites = []
                    splitted = match[0].split(",")
                    start = int(splitted[0])
                    end = start + len(splitted) - 1
                except ValueError:
                    raise CrawlerError("cite is no number")
                if cite_number == end + 1:
                    if book == 24:
                        break
                    book += 1
                    cite_number = 1
                    continue
                cite_number = end + 1
            else:
                raise CrawlerError("No cites declared in warning")
            lines = message.text.strip()[len(warning.strip()):]
            print("\n", book, start, end)
            print(lines.strip())
        # find_single_line_in_bodys(soup)
    else:
        print(r.status_code)
