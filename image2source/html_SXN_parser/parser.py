"""
Python file to convert HTML to SXN and vice versa
+ Currently, only tags related to design supported. <script> for example is not going to be parsed

I had the idea to put all the style in the corresponding element in the HTML. However, this will make the program to complicated and inefficient to parse.
I think we can just make two network, one is for HTML and another is for CSS. This is still in hypothesis step,
what I will do first, I will instead implement one network with a limited windows size. Hopefully this is enough for first prototype.


Author: Samuel Koesnadi 2019
"""
import logging

import bs4
from bs4 import BeautifulSoup
import re

bootstrap_link = """
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" crossorigin="anonymous">
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap-theme.min.css" integrity="sha384-rHyoN1iRsVXV4nD0JutlnGaslCJuC7uwjduW9SVrLvRYooPp2bWYgmgJQIXwl/Sp" crossorigin="anonymous">
"""
img_default_path_token = "***"
img_default_path = "../images/0.png"
default_url = "#"
css_open = '$'  # this cannot be used everywhere in CSS (reserved)
css_close = '&'  # this cannot be used everywhere in CSS (reserved)
css_space = '`'  #reserved


def removeComments(string):
    string = re.sub(re.compile("/\*.*?\*/",re.DOTALL ) ,"" ,string) # remove all occurrences streamed comments (/*COMMENT */) from string
    string = re.sub(re.compile("//.*?\n" ) ,"" ,string) # remove all occurrence single-line comments (//COMMENT\n ) from string
    return string

def encode_2_sxn(html):
    soup = BeautifulSoup(html, 'lxml')

    # TODO: if the seperator token is found that change it to something else
    def __encode_sxn(parent_elem):
        sxn_text = ""

        # check if it is a tag or not
        if parent_elem.name is None:
            text = parent_elem.strip()
            if text != "":
                # if this is style, then change

                # TODO: delete these lines of code, these are only necessary for pix2code because the text in html of the datasets
                #  do not match the text in the image
                text = " ".join(map(lambda _ : "texts", text.split(' ')))

                sxn_text += 't { ' + text + ' } '  # repr can be replaced to str as will for encoding
        else:
            sxn_text += parent_elem.name
            if len(parent_elem.attrs) != 0: # add attributes to sxn_text
                for key, val in parent_elem.attrs.items():
                    sxn_text += ' '+key+" = \" "

                    # switch case key::: you can put default value instead of what available
                    if key == "href" or key == "action":
                        sxn_text += default_url
                    elif key == "src":
                        sxn_text += img_default_path_token
                    elif key == "srcset":
                        pass
                    else:
                        sxn_text += ' '.join(val) if type(val) == list else val

                    sxn_text += " \""

            sxn_text += ' { '

            if parent_elem.name == "html":  # encode body before head, because I think it is easier for AI to write the body before the head
                for target_tagName in ["body", "head"]:
                    for elem in parent_elem.find_all(target_tagName):
                        sxn_text += __encode_sxn(elem)
            elif parent_elem.name == "head":  # exception for HEAD, only logging.info title and style if exists
                for target_tagName in ["title", "style"]:
                    for elem in parent_elem.find_all(target_tagName):
                        sxn_text += __encode_sxn(elem)
            else:
                # else encode all children tags
                if parent_elem.name == "style":
                    css = re.sub(r'}', css_close, re.sub(r'{', css_open, removeComments(parent_elem.string.strip())))
                    css = re.sub(r"(?<=\w)\s+(?=[\.\#\w\*])", css_space, css)
                    sxn_text += css
                else:
                    for elem in parent_elem.children:
                        if type(elem) != bs4.element.Comment and elem.name != "script":  # do not put comment in SXN and do not put <script> in SXN
                            sxn_text += __encode_sxn(elem)
            sxn_text += ' } '

        return sxn_text

    # move header to the end
    header = soup.find("header")
    header.name = "head"
    soup.html.insert(len(soup.html.contents), header)

    # select codes inside real HTML tag to logging.info
    list_of_contents_length = list(map(len, map(str, soup.contents)))
    soup_index = list_of_contents_length.index(max(list_of_contents_length))  # index of which soup content to logging.info (the longest is definitely what we are up to, which is HTML code)

    sxn_result = __encode_sxn(soup.contents[soup_index])  # encode the html to sxn

    # change to default value for img_path
    sxn_result = re.sub(r'url\s*\(\s*\"([^\"]*)\"\s*\)', "url(\""+img_default_path_token+"\")", sxn_result)

    # put whitespace in between proper punctuation and number
    sxn_result = re.sub(r'(?<=\W)([0-9]+)([a-zA-Zäüöß]+)', r"\1 \2", sxn_result)
    sxn_result = re.sub(r'([\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~])', r' \1 ', sxn_result)

    # filter all kinds of whitespaces
    sxn_result = sxn_result.replace("\t", "")
    sxn_result = re.sub(r'\s+', ' ', sxn_result)  # if there are whitespaces in a sequence, just put it once as it is just the same in HTML

    return sxn_result[sxn_result.index("{")+1:sxn_result.rfind("}")]  # remove html bla bla bla{ } from the result

def decode_2_html(sxn_result):

    # unfilter whitespaces of punctuation
    sxn_result = re.sub(r'\s*([\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~])\s*', r'\1', sxn_result)
    sxn_result = re.sub(r'(?<=\W)([0-9]+)\s+([a-zA-Zäüöß]+)', r"\1\2", sxn_result)

    # re-append html tag
    sxn_result = "html{" + sxn_result + "}"
    sxn_result = re.sub(r'\s+[{]\s+', "{", sxn_result)
    sxn_result = re.sub(r'\s+[}]\s+', "}", sxn_result)

    def __decode_sxn(sxn_text):
        html_result = ""  # the html result
        ch_str = ""  # the text placeholder
        tag_name = ""

        i_ch = 0
        while i_ch < len(sxn_text):
        # for i_ch, ch in enumerate(sxn_text):
            ch = sxn_text[i_ch]
            if ch == '{':
                if tag_name == "":
                    tag_name = ch_str.strip()
                    if tag_name.strip() != 't': html_result += '<'+tag_name+'>'

                    if tag_name.strip()[:4] == "head":  # put all the dependecies here
                        html_result += bootstrap_link

                    child_i_ch, child_html_result = __decode_sxn(sxn_text[i_ch+1:])

                    if "style" in tag_name:  # parse CSS back to normal design. TODO please explore this thing and fix if there is issue
                        child_html_result = child_html_result.replace(css_open, "{").replace(css_close, "}").replace(css_space, " ")
                else:
                    child_i_ch, child_html_result = __decode_sxn(sxn_text[i_ch - len(ch_str):])
                    i_ch -= len(ch_str) + 1
                html_result += child_html_result.strip()
                i_ch += child_i_ch

                # reset placeholder
                ch_str = ""
            elif ch == '}':
                break;
            else:
                ch_str += ch

            i_ch += 1  # add one to i_ch

        # I put it outside instead of in the elif ch == '}', so it can automatically close the tag
        if tag_name != "":
            if tag_name != 't':
                space_i = tag_name.find(' ')  # index of space found
                tag_name = tag_name if space_i == -1 else tag_name[:space_i]
                html_result += '</' + tag_name + '>'
            i_ch += 1  # this is to account for '{' that is skipped when it is decoded
        else:
            html_result = ch_str.strip()

        # if it is already touches the end then just return the ch_str
        return i_ch, html_result

    html_result = __decode_sxn(sxn_result)[1]

    # decode image path token with real image path
    html_result = html_result.replace(img_default_path_token, img_default_path)

    return BeautifulSoup(html_result, "lxml").prettify()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    # open the html file
    # filename = "all_data/0CE73E18-575A-4A70-9E40-F000B250344F.html"
    filename = "./assets/ground_truth_0.html"
    filename_generated = "generated_example.html"
    with open(filename, "r") as f:
        html = f.read()

    # logging.info the length of the original HTML file
    print("Original HTML length", len(html))

    # encode and logging.info the length and the html result
    sxn_result = encode_2_sxn(html)

    print("SXN length", len(sxn_result), repr(sxn_result))

    # decode and logging.info the html result of it
    html_result = decode_2_html(sxn_result)
    print("html_result:", len(html_result), repr(html_result))

    # store the generated html to file
    with open(filename_generated, "w") as f:
        f.write(html_result)
        logging.info("\n*** Generated HTML is stored to %s" % filename_generated)
