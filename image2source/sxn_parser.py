"""
Python file to convert HTML to SXN and vice versa
+ Currently, only tags related to design supported. <script> for example is not going to be parsed

I had the idea to put all the style in the corresponding element in the HTML. However, this will make the program to complicated and inefficient to parse.
I think we can just make two network, one is for HTML and another is for CSS. This is still in hypothesis step,
what I will do first, I will instead implement one network with a limited windows size. Hopefully this is enough for first prototype.


Author: Samuel Koesnadi 2019
"""
import os
import subprocess
from collections import defaultdict
from enum import Enum
import logging
from pathlib import Path

import bs4
from bs4 import BeautifulSoup
import re


class FileType(Enum):
    HTML = 0
    CSS = 1
    JS = 2

FILE_TYPES = {
    ".css": FileType.CSS,
    ".html": FileType.HTML
}


bootstrap_link = """
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">  
"""
img_default_path_token = "***"
img_default_path = "../images/0.png"
# default_url = "url"
css_open = '$'  # this cannot be used everywhere in CSS (reserved)
css_close = '&'  # this cannot be used everywhere in CSS (reserved)
css_space = '`'  # reserved

text_placeholder = "asdf"


def removeComments(string):
    # remove all occurrences streamed comments (/*COMMENT */) from string
    string = re.sub(re.compile("/\*.*?\*/",re.DOTALL ) ,"" ,string)

    # remove all occurrence single-line comments (//COMMENT\n ) from string
    string = re.sub(re.compile("//.*?\n" ) ,"" ,string)
    return string


def encode_css(elem_string):
    css = re.sub(r'}', css_close,
                 re.sub(r'{', css_open, removeComments(elem_string)))
    css = re.sub(r"(?<=\w)\s+(?=[\.\#\w\*])", css_space, css)

    return css


def decode_css(element):
    element = (
        element.replace(css_open, "{")
               .replace(css_close, "}")
               .replace(css_space, " "))

    return element


# TODO: if the seperator token is found that change it to something else
def __encode_sxn(parent_elem):
    sxn_text = ""

    # check if it is a tag or not
    if parent_elem.name is None:
        text = parent_elem.strip()
        if text != "":
            # TODO: delete these lines of code, these are only necessary for pix2code because the
            #  text in html of the datasets do not match the text in the image
            # text = " ".join(map(lambda _: text_placeholder, text.split(' ')))
            multiplier = len(text.split(' '))

            if multiplier > 1:
                text = f"{text_placeholder} {multiplier}"
            else:
                text = text_placeholder

            sxn_text += 't { ' + text + ' } '  # repr can be replaced to str as will for encoding
    else:
        sxn_text += parent_elem.name
        if len(parent_elem.attrs) != 0: # add attributes to sxn_text
            for key, val in parent_elem.attrs.items():
                sxn_text += ' ' + key + " = \" "

                # switch case key::: you can put default value instead of what available
                if key == "href" or key == "action":
                    sxn_text += val
                elif key == "src":
                    sxn_text += img_default_path_token
                elif key == "srcset":
                    pass
                else:
                    sxn_text += ' '.join(val) if type(val) == list else val
                sxn_text += " \""

        sxn_text += ' { '

        # encode body before head, because I think it is easier for AI to write the body
        # before the head
        if parent_elem.name == "html":
            for target_tagName in ["body", "head"]:
                for elem in parent_elem.find_all(target_tagName):
                    sxn_text += __encode_sxn(elem)
        # exception for HEAD, only logging.info title and style if exists
        elif parent_elem.name == "head":
            for target_tagName in ["title", "style", "link"]:
                for elem in parent_elem.find_all(target_tagName):
                    sxn_text += __encode_sxn(elem)
        else:
            # else encode all children tags
            if parent_elem.name == "style":
                css = encode_css(parent_elem.string.strip())
                sxn_text += css
            else:
                for elem in parent_elem.children:
                    # do not put comment in SXN and do not put <script> in SXN
                    # TODO!
                    if type(elem) != bs4.element.Comment and elem.name != "script":
                        sxn_text += __encode_sxn(elem)
        sxn_text += ' } '

    return sxn_text


def encode_2_sxn(html, type):
    if type == FileType.CSS:
        sxn_result = ' ' + encode_css(html)
    else:
        # Filter unnecessary tags
        # 1. filter bootstrap source
        html = re.sub(r"<.*bootstrap\.min\.css.*>", "", html)
        html = re.sub(r"<.*bootstrap-theme\.min\.css.*>", "", html)
        # 2. filter rel=canonical
        html = re.sub(r"<.*rel=\"canonical\".*>", "", html)

        soup = BeautifulSoup(html, 'lxml')

        # move header to the end
        headers = soup.find_all(["head", "header"])
        for header in headers:
            if header.parent.name == "html":
                header.name = "head"
                soup.html.insert(len(soup.html.contents), header)

        # select codes inside real HTML tag to logging.info
        list_of_contents_length = list(map(len, map(str, soup.contents)))
        soup_index = list_of_contents_length.index(max(list_of_contents_length))  # index of which soup content to logging.info (the longest is definitely what we are up to, which is HTML code)

        sxn_result = __encode_sxn(soup.contents[soup_index])  # encode the html to sxn

        # remove html bla bla bla{ } from the result
        sxn_result = sxn_result[sxn_result.index("{")+1:sxn_result.rfind("}")]

    # change to default value for img_path
    sxn_result = re.sub(r'url\s*\(\s*\"([^\"]*)\"\s*\)', "url(\""+img_default_path_token+"\")", sxn_result)

    # put whitespace in between proper punctuation and number

    # FIXME: I dont understand this
    # sxn_result = re.sub(r'(?<=\W)([0-9]+)([a-zA-Zäüöß]+)', r"\1 \2", sxn_result)

    sxn_result = re.sub(
        r'([\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~])',
        r' \1 ', sxn_result)

    # filter all kinds of whitespaces
    sxn_result = sxn_result.replace("\t", "")
    sxn_result = re.sub(r'\s+', ' ', sxn_result)  # if there are whitespaces in a sequence, just put it once as it is just the same in HTML

    return sxn_result


def __decode_sxn(sxn_text):
    html_result = ""  # the html result
    ch_str = ""  # the text placeholder
    tag_name = ""

    i_ch = 0
    while i_ch < len(sxn_text):
        ch = sxn_text[i_ch]
        if ch == '{':
            if tag_name == "":
                tag_name = ch_str.strip()
                if tag_name.strip() != 't':
                    html_result += '<' + tag_name + '>'

                if tag_name.strip()[:4] == "head":  # put all the dependencies here
                    html_result += bootstrap_link

                child_i_ch, child_html_result = __decode_sxn(sxn_text[i_ch + 1:])

                if "style" in tag_name:  # parse CSS back to normal design. TODO please explore this thing and fix if there is issue
                    child_html_result = decode_css(child_html_result)
            else:
                child_i_ch, child_html_result = __decode_sxn(sxn_text[i_ch - len(ch_str):])
                i_ch -= len(ch_str) + 1

            # put the content to the html
            child_html_result = child_html_result.strip()
            if child_html_result.startswith(text_placeholder):
                multiplier = child_html_result[len(text_placeholder) + 1:]

                if multiplier.isnumeric():
                    child_html_result = ' '.join([text_placeholder] * int(multiplier))

            html_result += child_html_result
            i_ch += child_i_ch

            # reset placeholder
            ch_str = ""
        elif ch == '}':
            break
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


def decode_2_html(sxn_result, type):
    # unfilter whitespaces of punctuation
    sxn_result = re.sub(r'\s*([\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~])\s*', r'\1', sxn_result)

    # FIXME: check if this necessary
    # sxn_result = re.sub(r'(?<=\W)([0-9]+)\s+([a-zA-Zäüöß]+)', r"\1\2", sxn_result)

    if type == FileType.CSS:
        html_result = decode_css(sxn_result)
    else:
        # re-append html tag
        sxn_result = "html{" + sxn_result + "}"
        sxn_result = re.sub(r'\s+[{]\s+', "{", sxn_result)
        sxn_result = re.sub(r'\s+[}]\s+', "}", sxn_result)
        html_result = __decode_sxn(sxn_result)[1]

        # decode image path token with real image path
        html_result = html_result.replace(img_default_path_token, img_default_path)

        # html_result = BeautifulSoup(html_result, "lxml").prettify()

    return html_result


def encode_sxn_folder(folder_to_convert, target_dir):
    onlyfiles = [f for f in os.listdir(folder_to_convert) if os.path.isfile(os.path.join(folder_to_convert, f))]

    # make directory if does not exist
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    files_sxn = defaultdict(list)
    for file in onlyfiles:
        filename, file_extension = os.path.splitext(file)

        if file_extension not in FILE_TYPES:
            continue

        file_type = FILE_TYPES[file_extension]

        # read the content
        with open(os.path.join(folder_to_convert, file), "r") as f:
            content = f.read()

        sxn_result = encode_2_sxn(content, file_type)

        if file_type == FileType.HTML:
            filename_tag = "html"
        else:
            filename_tag = ' . '.join(file.split('.'))

        files_sxn[file_type].append((f'{filename_tag} <content>' + sxn_result, file))

    # combine all CSS
    all_css_sxn = ' <eop> '.join([sxn.strip() for sxn, _ in files_sxn[FileType.CSS]])

    # match every HTML with all the CSS
    website_dict = {}
    for html_sxn, ori_html_file_path in files_sxn[FileType.HTML]:
        new_uuid = (
            folder_to_convert + ' ' + ori_html_file_path).replace(' ', '-').replace('.', '-').replace('/', '-')
        if all_css_sxn == '':
            complete_sxn = html_sxn
        else:
            complete_sxn = ' '.join([html_sxn.strip(), "<eop>", all_css_sxn])

        # # add end-of-document to the end of the text
        # complete_sxn += " <eod>"

        with open(os.path.join(target_dir, f'{new_uuid}.sxn'), 'w') as f:
            f.write(complete_sxn)

        # take screenshot for image
        screenshot_command = (
            f"node headless_screenshot/screenshot.js"
            f" --url file://{os.path.abspath(os.path.join(folder_to_convert, ori_html_file_path))}"
            f" --fullPage --targetFilePath {os.path.abspath(os.path.join(target_dir, f'{new_uuid}.png'))}")
        subprocess.run(screenshot_command.split())

        website_dict[str(new_uuid)] = len(complete_sxn.split(' '))

    return website_dict  # the file length dictionary


def decode_sxn_folder(sxn_string):
    results = {}

    splitted_sxn = sxn_string.split()

    for i_index_content in (
            [i for i, x in enumerate(splitted_sxn) if x == "<content>"]
    ):
        # get the file type
        parsed_file_type = None
        parsed_file_name = None
        if splitted_sxn[i_index_content - 1] == "html":
            parsed_file_type = FileType.HTML
            parsed_file_name = "index.html"
        elif splitted_sxn[i_index_content - 1] == "css":
            parsed_file_type = FileType.CSS

            flipped_parsed_file_name = []
            for i_look_file_name in range(i_index_content - 1, i_index_content - 5, -1):  # maximum 5 characters before
                parsed_token = splitted_sxn[i_look_file_name]
                if parsed_token == "<eop>":  # break if find the <eop> of previous file
                    break

                flipped_parsed_file_name.append(parsed_token)
            parsed_file_name = ''.join(flipped_parsed_file_name[::-1])

        # parse the content
        cut_parsed_content = splitted_sxn[i_index_content + 1:]
        if "<eop>" in cut_parsed_content:
            i_cut_parsed_content_found = splitted_sxn.index("<eop>")
            cut_parsed_content = cut_parsed_content[:i_cut_parsed_content_found]

        # store the result in dictionary
        results[parsed_file_name] = decode_2_html(
            " ".join(cut_parsed_content), parsed_file_type)

    return results


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    # with open("/media/radoxpi/Garage/project/mockup_ai/datasets/parsed/datasets-raw-black_and_white-another_page-html.sxn") as f:
    #     results = decode_sxn_folder(f.read())

    sxn_string = """html <content> body { div id = " main " { div id = " header " { div id = " logo " { div id = " logo _ text " { h1 { a href = " index . html " { t { asdf } span class = " logo _ colour " { t { asdf } } } } h2 { t { asdf 4 } } } } div id = " menubar " { ul id = " menu " { li { a href = " index . html " { t { asdf } } } li class = " selected " { a href = " examples . html " { t { asdf } } } li { a href = " page . html " { t { asdf 2 } } } li { a href = " another _ page . html " { t { asdf 2 } } } li { a href = " contact . html " { t { asdf 2 } } } } } } div id = " content _ header " { } div id = " site _ content " { div class = " sidebar " { h3 { t { asdf 2 } } h4 { t { asdf 3 } } h5 { t { asdf 3 } } p { t { asdf 18 } br { } a href = " # " { t { asdf 2 } } } p { } h4 { t { asdf 3 } } h5 { t { asdf 3 } } p { t { asdf 18 } br { } a href = " # " { t { asdf 2 } } } h3 { t { asdf 2 } } ul { li { a href = " # " { t { asdf 2 } } } li { a href = " # " { t { asdf 2 } } } li { a href = " # " { t { asdf 2 } } } li { a href = " # " { t { asdf 2 } } } } h3 { t { asdf } } form method = " post " action = " # " id = " search _ form " { p { input class = " search " type = " text " name = " search _ field " value = " enter keywords . . . . . " { } input name = " search " type = " image " style = " border : 0 ; margin : 0 0 - 9px 5px ; " src = " * * * " alt = " search " title = " search " { } } } } div id = " content " { h1 { t { asdf 2 } p { t { asdf 25 } } h2 { t { asdf } } p { asdf 2 } } h2 { t { t { t { asdf 62 } } p { t { asdf 62 } } p { t { asdf 62 } } } } div id = " content _ footer " { } div id = " footer " { t { asdf 2 } a href = " http : / / validator . w3 . org / check ? uri = referer " { t { asdf } } t { asdf } a href = " http : / / jigsaw . w3 . org / css - validator / check / referer " { t { asdf } } t { asdf } a href = " http : / / www . html5webtemplates . co . uk " { t { asdf 3 } } } } } head { title { t { asdf 3 } } link rel = " stylesheet " type = " text / css " href = " style . css " title = " style " { } } <eop> style . css <content> html $ height : 100 % ; & * $ margin : 0 ; padding : 0 ; & body $ font : normal ` . 80em \' trebuchet ` ms \' , arial , sans - serif ; background : # f0efe2 ; & p $ font : 0 ` 0 ` 20px ` 0 ; & h1 , h2 , h3 , h5 , h3 , h2 , h3 , h4 , h5 , h6 $ font : normal ` 175 % \' century ` gothic \' , arial , sans - serif ; & h1 , sans - serif ; color : # fff ; margin : 0 ` 0 ` 15px ` 0 ; padding : 15px ` 0 ` 5px ` 0 ; & h2 $ font : normal ` 175 % \' century ` 175 % \' century ` gothic \' , arial , sans - serif ; & h4 , sans - serif ; background : 0 ; padding : 0 ` 0 ` 5px ` 0 ; & h5 , h6 $ font : normal ` 120 % arial , sans - serif ; & h4 , h6 $ font : italic ` 95 % arial , sans - serif ; padding : 0 ` 0 ` 95 % arial , sans - height : # 362c20 ; & h2 , h6 $ outline : none ; & . left $ float : none ; margin : auto ; margin - right : 10px ; & . right $ display : right : 10px ; & . center $ display : block ; & . center $ display : block ; text - align : center ; margin : 20px ` auto ; & . center $ display : 20px ` 20px ` 0 ; margin : 20px ` 0 ` 0 ` 20px ` 17px ; & ul ` 0 ` 0 ` 0 ` auto ; & blockquote $ margin : 20px ` 0 ` 0 ` solid ` 20px ` 0 ` 20px ; border : 0 ` solid ` # e5e5db ; background : # fff ; & ul $ margin : 20px ; border : 0 ` 17px ; & ul ` li $ list - style - type : circle ; & # menubar , # site _ content , # footer $ margin - left : auto ; margin : 20px ` 0 ; border - align : 20px ` 0 ` # e5e5db ; border - left : 20px ` 0 ` 0 ` 0 ` 0 ` 0 ` 0 ` 0 ` 0 ` 0 ` 0 ` 0 ` 0 ` 0 ` 0 ` 0 ` 0 ; & # logo _ content , # header $ background : # footer $ margin : 0 ` 0 ` 0 ` 0 ; border : 0 ; & # footer $ position : 20px ; & # header $ background : 0 ; & # logo _ text ` 0 ` 0 ` 0 ` 0 ` 0 ` url ( back . center ; position : 0 ; & # logo ` h1 , # logo ` h2 $ font : 0 ; & # logo _ text ` 0 ` 0 ` 0 ` h1 ` . png ) repeat - align : 0 ; & # logo $ position : center ; & # site _ content , # logo _ text - bottom : # logo _ text $ position : # logo _ text ` 0 ` h1 , # logo ` h2 $ font : # logo _ text ` h2 $ font - top : # logo _ text - bottom : 0 ` 0 ` 0 ; height : # fff ; & # logo _ colour $ color : # header , # logo _ text ` 300 % \' century ` 0 ` url ( back . png ) repeat - size : auto ; & # logo _ text - bottom : 0 ; & # logo _ text - bottom : 0 ` a , # logo _ text ` url ( menu . png ) repeat - x ; & ul # logo _ text ` h1 ` a ` 0 ` a ` h2 $ font : 0 ; text ` a ` 0 ; & # logo _ text ` 0 ` # logo _ text ` 0 ; & # logo _ text ` 0 ` . 1em ; font : normal ` 100 % \' lucida ` sans ` 300 % \' century ` a ` . logo _ colour $ color : # fff ; height : 100 % ; padding : # fff ; & ul # menu ` 0 ; & # fff ; & # fff ; & ul # fff ; & # menubar $ width : 0"""
    results = decode_sxn_folder(sxn_string)
    for key, value in results.items():
        with open(key, 'w') as f:
            f.write(value)
    exit()

    # open the html file
    # filename = "all_data/0CE73E18-575A-4A70-9E40-F000B250344F.html"

    # filename = "./assets/ground_truth_0.html"
    # filename_generated = "generated_example.html"
    #
    # folder_to_convert = "/media/radoxpi/Garage/project/mockup_ai/datasets/raw/features"
    #
    # encode_sxn_folder(folder_to_convert, "datasets/parsed")




    filename = "./datasets/raw/pix2code/0B660875-60B4-4E65-9793-3C7EB6C8AFD0.html"
    filename_generated = (
        "/media/radoxpi/Garage/project/mockup_ai/datasets/raw/features/generated_example.html")

    name, extension = os.path.splitext(filename)

    file_type = FILE_TYPES[extension]

    with open(filename, "r") as f:
        html = f.read()

    # logging.info the length of the original HTML file
    print("Original HTML length", len(html))

    # encode and logging.info the length and the html result
    sxn_result = encode_2_sxn(html, file_type)

    print("SXN length", len(sxn_result.split()), repr(sxn_result))

    # decode and logging.info the html result of it
    html_result = decode_2_html(sxn_result, file_type)
    print("html_result:", len(html_result), repr(html_result))

    # store the generated html to file
    with open(filename_generated, "w") as f:
        f.write(html_result)
        logging.info("\n*** Generated HTML is stored to %s" % filename_generated)
