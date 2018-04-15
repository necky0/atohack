import urllib.request as urllib


def get_response(url):
    response = urllib.urlopen(url)
    return response.read().decode().splitlines()


def extract_links(html):
    lines = []
    for line in [l for l in html if l.find('o-list_item_link_name') >= 0]:
        lines.append(line)

    return lines


def job_list(category, city):
    url = get_url(category, city)
    html = get_response(url)
    lines = extract_links(html)

    data = []
    for line in lines:
        url_marker = '<a href="/praca'
        index_from = line.find(url_marker) + len(url_marker)
        url = line[index_from:]
        index_to = url.find('"')
        url = url[:index_to]

        title_marker = 'title="'
        index_from = line.find(title_marker) + len(title_marker)
        title = line[index_from:]
        index_from = title.find('>') + 1
        title = title[index_from:]
        index_to = title.find('</a>')
        title = title[:index_to]

        data.append((title, 'https://www.pracuj.pl/praca' + url))

    return data


def get_url(category, city):
    return 'https://www.pracuj.pl/praca/{};kw/{};wp'.format(category, city)
