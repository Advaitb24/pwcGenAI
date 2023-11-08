import asyncio
import streamlit as st
import prompts
import model
import storage
import cache
import os

from time import time as now

import Components.StreamlitSetup as StreamlitSetup
StreamlitSetup.setup()

from Components.sidebar import sidebar
import Modules.file_io as file_io
import GPT
import util
import time
from streamlit_option_menu import option_menu
from PyPDF2 import PdfReader
import openai
from langchain.text_splitter import CharacterTextSplitter

ss = st.session_state
if 'debug' not in ss: ss['debug'] = {}
import css
st.write(f'<style>{css.v1}</style>', unsafe_allow_html=True)
header1 = st.empty() # for errors / messages
header2 = st.empty() # for errors / messages
header3 = st.empty() # for errors / messages

app_header = st.container()

file_handler = st.container()
content_handler = st.container()
result_handler = st.container()

def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    chunks = ""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=30,
        length_function=len
    )
    for page in pdf_reader.pages:
        text = page.extract_text()

        chunks = text_splitter.split_text(text)
    return chunks

def compare_text_with_gpt4(text1, text2):
	
    prompt = f"Compare the following two texts:\n{file1.name}:\n{text1}\n{file2.name}:\n{text2}\n eloborate Differences: , also show differnce in each clauses if any"
    text1 = ' '.join([str(elem) for elem in text1])
    text2 = ' '.join([str(elem) for elem in text2])
    response = openai.Completion.create(
        engine="text-davinci-003",  # You may need to use the appropriate engine here
        prompt=prompt,
        temperature=0.1,
        max_tokens=1000  # Adjust as needed for your use case
    )
	
    generated_text =response
    return generated_text
    comparison_result = response.choices[0].text
    return comparison_result

def on_api_key_change():
	api_key = ss.get('api_key') or os.getenv('OPENAI_KEY')
	model.use_key(api_key) # TODO: empty api_key
	#
	if 'data_dict' not in ss: ss['data_dict'] = {} # used only with DictStorage
	ss['storage'] = storage.get_storage(api_key, data_dict=ss['data_dict'])
	ss['cache'] = cache.get_cache()
	ss['user'] = ss['storage'].folder # TODO: refactor user 'calculation' from get_storage
	model.set_user(ss['user'])
	# ss['feedback'] = feedback.get_feedback_adapter(ss['user'])
	# ss['feedback_score'] = ss['feedback'].get_score()
	#
	ss['debug']['storage.folder'] = ss['storage'].folder
	ss['debug']['storage.class'] = ss['storage'].__class__.__name__


ss['community_user'] = os.getenv('COMMUNITY_USER')
if 'user' not in ss and ss['community_user']:
	on_api_key_change() # use community key

# COMPONENTS


def ui_spacer(n=2, line=False, next_n=0):
	for _ in range(n):
		st.write('')
	if line:
		st.tabs([' '])
	for _ in range(next_n):
		st.write('')


def ui_api_key():
	if ss['community_user']:
		st.write('## 1. Optional - enter your OpenAI API key')
		t1,t2 = st.tabs(['community version','enter your own API key'])
		with t1:
			pct = model.community_tokens_available_pct()
			st.write(f'Community tokens available: :{"green" if pct else "red"}[{int(pct)}%]')
			st.progress(pct/100)
			st.write('Refresh in: ' + model.community_tokens_refresh_in())
			st.write('You can sign up to OpenAI and/or create your API key [here](https://platform.openai.com/account/api-keys)')
			ss['community_pct'] = pct
			ss['debug']['community_pct'] = pct
		with t2:
			st.text_input('OpenAI API key', type='password', key='api_key', on_change=on_api_key_change, label_visibility="collapsed")
	else:
		st.write('## 1. Enter your OpenAI API key')
		st.text_input('OpenAI API key', type='password', key='api_key', on_change=on_api_key_change, label_visibility="collapsed")

def index_pdf_file():
	if ss['pdf_file']:
		ss['filename'] = ss['pdf_file'].name
		if ss['filename'] != ss.get('fielname_done'): # UGLY
			with st.spinner(f'indexing {ss["filename"]}'):
				index = model.index_file(ss['pdf_file'], ss['filename'], fix_text=ss['fix_text'], frag_size=ss['frag_size'], cache=ss['cache'])
				ss['index'] = index
				debug_index()
				ss['filename_done'] = ss['filename'] # UGLY

def debug_index():
	index = ss['index']
	d = {}
	d['hash'] = index['hash']
	d['frag_size'] = index['frag_size']
	d['n_pages'] = len(index['pages'])
	d['n_texts'] = len(index['texts'])
	d['summary'] = index['summary']
	d['pages'] = index['pages']
	d['texts'] = index['texts']
	d['time'] = index.get('time',{})
	ss['debug']['index'] = d

def ui_pdf_file():
	st.write('## 2. Upload or select your PDF file')
	disabled = not ss.get('user') or (not ss.get('api_key') and not ss.get('community_pct',0))
	t1,t2 = st.tabs(['UPLOAD','SELECT'])
	with t1:
		st.file_uploader('pdf file', type='pdf', key='pdf_file', disabled=disabled, on_change=index_pdf_file, label_visibility="collapsed")
		# b_save()
	with t2:
		filenames = ['']
		if ss.get('storage'):
			filenames += ss['storage'].list()
		def on_change():
			name = ss['selected_file']
			if name and ss.get('storage'):
				with ss['spin_select_file']:
					with st.spinner('loading index'):
						t0 = now()
						index = ss['storage'].get(name)
						ss['debug']['storage_get_time'] = now()-t0
				ss['filename'] = name # XXX
				ss['index'] = index
				debug_index()
			else:
				#ss['index'] = {}
				pass
		st.selectbox('select file', filenames, on_change=on_change, key='selected_file', label_visibility="collapsed", disabled=disabled)
		b_delete()
		ss['spin_select_file'] = st.empty()

def ui_show_debug():
	st.checkbox('show debug section', key='show_debug')

def ui_fix_text():
	st.checkbox('fix common PDF problems', value=True, key='fix_text')

def ui_temperature():
	#st.slider('temperature', 0.0, 1.0, 0.0, 0.1, key='temperature', format='%0.1f')
	ss['temperature'] = 0.0

def ui_fragments():
	#st.number_input('fragment size', 0,2000,200, step=100, key='frag_size')
	st.selectbox('fragment size (characters)', [0,200,300,400,500,600,700,800,900,1000], index=3, key='frag_size')
	b_reindex()
	st.number_input('max fragments', 1, 10, 4, key='max_frags')
	st.number_input('fragments before', 0, 3, 1, key='n_frag_before') # TODO: pass to model
	st.number_input('fragments after',  0, 3, 1, key='n_frag_after')  # TODO: pass to model

def ui_model():
	models = ['gpt-3.5-turbo','gpt-4','text-davinci-003','text-curie-001']
	st.selectbox('main model', models, key='model', disabled=not ss.get('api_key'))
	st.selectbox('embedding model', ['text-embedding-ada-002'], key='model_embed') # FOR FUTURE USE

def ui_hyde():
	st.checkbox('use HyDE', value=True, key='use_hyde')

def ui_hyde_summary():
	st.checkbox('use summary in HyDE', value=True, key='use_hyde_summary')

def ui_task_template():
	st.selectbox('task prompt template', prompts.TASK.keys(), key='task_name')

def ui_task():
	x = ss['task_name']
	st.text_area('task prompt', prompts.TASK[x], key='task')

def ui_hyde_prompt():
	st.text_area('HyDE prompt', prompts.HYDE, key='hyde_prompt')

def ui_question():
	st.write('## 3. Ask questions'+(f' to {ss["filename"]}' if ss.get('filename') else ''))
	disabled = False
	st.text_area('question', key='question', height=100, placeholder='Enter question here', help='', label_visibility="collapsed", disabled=disabled)

# REF: Hypotetical Document Embeddings
def ui_hyde_answer():
	# TODO: enter or generate
	pass

def ui_output():
	output = ss.get('output','')
	st.markdown(output)

def ui_debug():
	if ss.get('show_debug'):
		st.write('### debug')
		st.write(ss.get('debug',{}))


def b_ask():
	c1,c2,c3,c4,c5 = st.columns([2,1,1,2,2])
	# if c2.button('👍', use_container_width=True, disabled=not ss.get('output')):
	# 	ss['feedback'].send(+1, ss, details=ss['send_details'])
	# 	ss['feedback_score'] = ss['feedback'].get_score()
	# if c3.button('👎', use_container_width=True, disabled=not ss.get('output')):
	# 	ss['feedback'].send(-1, ss, details=ss['send_details'])
	# 	ss['feedback_score'] = ss['feedback'].get_score()
	# score = ss.get('feedback_score',0)
	# c5.write(f'feedback score: {score}')
	# c4.checkbox('send details', True, key='send_details',
	# 		help='allow question and the answer to be stored in the ask-my-pdf feedback database')
	#c1,c2,c3 = st.columns([1,3,1])
	#c2.radio('zzz',['👍',r'...',r'👎'],horizontal=True,label_visibility="collapsed")
	#
	disabled = (not ss.get('api_key') and not ss.get('community_pct',0)) or not ss.get('index')
	if c1.button('get answer', disabled=disabled, type='primary', use_container_width=True):
		question = ss.get('question','')
		temperature = ss.get('temperature', 0.0)
		hyde = ss.get('use_hyde')
		hyde_prompt = ss.get('hyde_prompt')
		if ss.get('use_hyde_summary'):
			summary = ss['index']['summary']
			hyde_prompt += f" Context: {summary}\n\n"
		task = ss.get('task')
		max_frags = ss.get('max_frags',1)
		n_before = ss.get('n_frag_before',0)
		n_after  = ss.get('n_frag_after',0)
		index = ss.get('index',{})
		with st.spinner('preparing answer'):
			resp = model.query(question, index,
					task=task,
					temperature=temperature,
					hyde=hyde,
					hyde_prompt=hyde_prompt,
					max_frags=max_frags,
					limit=max_frags+2,
					n_before=n_before,
					n_after=n_after,
					model=ss['model'],
				)
		usage = resp.get('usage',{})
		usage['cnt'] = 1
		ss['debug']['model.query.resp'] = resp
		ss['debug']['resp.usage'] = usage
		ss['debug']['model.vector_query_time'] = resp['vector_query_time']
		
		q = question.strip()
		a = resp['text'].strip()
		ss['answer'] = a
		output_add(q,a)
		st.experimental_rerun() # to enable the feedback buttons

def b_clear():
	if st.button('clear output'):
		ss['output'] = ''

def b_reindex():
	# TODO: disabled
	if st.button('reindex'):
		index_pdf_file()

def b_reload():
	if st.button('reload prompts'):
		import importlib
		importlib.reload(prompts)

def b_save():
	db = ss.get('storage')
	index = ss.get('index')
	name = ss.get('filename')
	api_key = ss.get('api_key')
	disabled = not api_key or not db or not index or not name
	help = "The file will be stored for about 90 days. Available only when using your own API key."
	if st.button('save encrypted index in ask-my-pdf', disabled=disabled, help=help):
		with st.spinner('saving to ask-my-pdf'):
			db.put(name, index)

def b_delete():
	db = ss.get('storage')
	name = ss.get('selected_file')
	# TODO: confirm delete
	if st.button('delete from ask-my-pdf', disabled=not db or not name):
		with st.spinner('deleting from ask-my-pdf'):
			db.delete(name)
		#st.experimental_rerun()

def output_add(q,a):
	if 'output' not in ss: ss['output'] = ''
	q = q.replace('$',r'\$')
	a = a.replace('$',r'\$')
	new = f'#### {q}\n{a}\n\n'
	ss['output'] = new + ss['output']


with app_header:
    #   st.set_page_config(layout='wide')

    # PwC Contract Insights , Compare Your Contracts
    selected = option_menu(
        menu_title=None,
        options=["PDF QnA","PDF Summarizer"],
        icons=["book","book","envelope"], 
        default_index=0,
        orientation="horizontal"
    )
    if selected =="PDF Summarizer":
        st.title("Contract Insights summarizer")
    
        sidebar()

        with file_handler:
            if st.button("🔃 Refresh"):
                st.cache_data.clear()
            youtube_link_empty = st.empty()
            upload_file_emtpy = st.empty()
            youtube_link=""
            # youtube_link = youtube_link_empty.text_input(label="🔗 YouTube Link",
            #                                              placeholder="Enter your YouTube link",
            #                                              help="Enter your YouTube link to download the video and extract the audio")
            upload_file = upload_file_emtpy.file_uploader("📁 Upload your file", type=['txt', 'pdf', 'docx', 'md'])
            if youtube_link:
                upload_file_emtpy.empty()
                with st.spinner("🔍 Extracting transcript..."):
                    transcript, title = Modules.Youtube.extract_youtube_transcript(youtube_link, st.session_state['CAPTION_LANGUAGES'])
                    file_content = {'name': f"{title}.txt", 'content': transcript}
            elif upload_file:
                youtube_link_empty.empty()
                with st.spinner("🔍 Reading file... (mp3 file might take a while)"):
                    file_content = {'name': upload_file.name, 'content': file_io.read(upload_file)}
            elif youtube_link and upload_file:
                st.warning("Please only upload one file at a time")
            else:
                file_content = None

        with content_handler:
            if file_content:
                with st.expander("File Preview"):
                    if file_content['name'].endswith(".pdf"):
                        content = "\n\n".join(file_content['content'])
                        st.text_area(file_content['name'], content, height=200)
                    else:
                        content = file_content['content']
                        st.text_area(file_content['name'], content, height=200)

        with result_handler:
            if file_content:
                chunks = []
                content = file_content['content']
                if file_content['name'].endswith(".pdf"):
                    content = "\n\n".join(file_content['content'])
                chunks.extend(util.convert_to_chunks(content, chunk_size=st.session_state['CHUNK_SIZE']))

                with st.expander(f"Chunks ({len(chunks)})"):
                    for chunk in chunks:
                        st.write(chunk)

                token_usage = GPT.misc.predict_token(st.session_state['OPENAI_PARAMS'], chunks)
                param = st.session_state["OPENAI_PARAMS"]
                prompt_token = token_usage['prompt']
                completion_token = token_usage['completion']
                if param.model == 'gpt-4':
                    price = round(prompt_token * 0.00003 + completion_token * 0.00006, 5)
                    st.markdown('**Note:** To access GPT-4, please [join the waitlist](https://openai.com/waitlist/gpt-4-api)'
                                " if you haven't already received an invitation from OpenAI.")
                    st.info("ℹ️️ Please keep in mind that GPT-4 is significantly **[more expensive](https://openai.com/pricing#language-models)** than GPT-3.5. ")
                elif param.model == 'gpt-3.5-turbo-16k':
                    price = round(prompt_token * 0.000003 + completion_token *0.000004, 5)
                else:
                    price = round(prompt_token * 0.0000015 + completion_token * 0.000002 , 5)
                st.markdown(
                    f"Price Prediction: `${price}` || Total Prompt: `{prompt_token}`, Total Completion: `{completion_token}`")
                # max tokens exceeded warning
                exceeded = util.exceeded_token_handler(param=st.session_state['OPENAI_PARAMS'], chunks=chunks)

                # load cached results
                if st.session_state['PREVIOUS_RESULTS'] is not None:
                    rec_responses = st.session_state['PREVIOUS_RESULTS']['rec_responses']
                    rec_id = st.session_state['PREVIOUS_RESULTS']['rec_ids']
                    final_response = st.session_state['PREVIOUS_RESULTS']['final_response']
                    finish_reason_rec = st.session_state['PREVIOUS_RESULTS']['finish_reason_rec']
                    finish_reason_final = st.session_state['PREVIOUS_RESULTS']['finish_reason_final']
                else:
                    rec_responses = None
                    rec_id = None
                    final_response = None
                    finish_reason_rec = None
                    finish_reason_final = None

                # finish_reason_rec = None
                if st.button("🚀 Run", disabled=exceeded):
                    start_time = time.time()
                    st.cache_data.clear()
                    API_KEY = st.session_state['OPENAI_API_KEY']
                    if not API_KEY and not GPT.misc.validate_api_key(API_KEY):
                        st.error("❌ Please enter a valid [OpenAI API key](https://beta.openai.com/account/api-keys).")
                    else:
                        with st.spinner("Summarizing... (this might take a while)"):
                            if st.session_state['LEGACY']:
                                rec_max_token = st.session_state['OPENAI_PARAMS'].max_tokens_rec
                                rec_responses, finish_reason_rec = util.recursive_summarize(chunks, rec_max_token)
                                if st.session_state['FINAL_SUMMARY_MODE']:
                                    final_response, finish_reason_final = util.summarize(rec_responses)
                                else:
                                    final_response = None
                            else:
                                completions, final_response = asyncio.run(util.summarize_experimental_concurrently(content, st.session_state['CHUNK_SIZE']))
                                rec_responses = [d["content"] for d in completions]
                                rec_ids = [d["chunk_id"] for d in completions]
                                # save previous completions
                                resp = {'rec_responses': rec_responses,
                                        'rec_ids': rec_ids,
                                        'final_response': final_response,
                                        'finish_reason_rec': finish_reason_rec,
                                        'finish_reason_final': finish_reason_final}
                                if resp != st.session_state['PREVIOUS_RESULTS']:
                                    st.session_state['PREVIOUS_RESULTS'] = resp

                    end_time = time.time()
                    st.markdown(f"⏱️ Time taken: `{round(end_time - start_time, 2)}s`")

                if rec_responses is not None:
                    with st.expander("Recursive Summaries", expanded=not st.session_state['FINAL_SUMMARY_MODE']):
                        for i, response in enumerate(rec_responses):
                            st.info(f'{response}')
                    if finish_reason_rec == 'length':
                        st.warning('⚠️Result cut off due to length. Consider increasing the [Max Tokens Chunks] parameter.')

                if final_response is not None:
                    st.header("📝Summary")
                    st.info(final_response)
                    if finish_reason_final == 'length':
                        st.warning(
                            '⚠️Result cut off due to length. Consider increasing the [Max Tokens Summary] parameter.')
                if final_response is not None or rec_responses is not None:
                    util.download_results(rec_responses, final_response)

    if selected=="PDF Comparator":
        # Streamlit app
        st.title("PwC Contract Insights , Compare Your Contracts")

        # Upload two PDF files
        st.sidebar.title("Upload PDF Files")
        file1 = st.sidebar.file_uploader("Upload the first PDF file", type=["pdf"])
        file2 = st.sidebar.file_uploader("Upload the second PDF file", type=["pdf"])

        # Compare button
        if st.sidebar.button("Compare"):
            if file1 is not None and file2 is not None:
                st.subheader("Comparison Result")
                
                # Extract text from both PDFs
                text1 = extract_text_from_pdf(file1)
                text2 = extract_text_from_pdf(file2)
                                # prompt=f"Compare the following two texts:\n{file1.name}: {text1}\n{file2.name}: {text2}\n eloborate Differences: , also show differnce in each clauses if any"
                # prompt = f"Compare the following two texts:\n{file1.name}:\n{text1}\n{file2.name}:\n{text2}\n provide exact term differences in between both the contracts:"
                comparison_result = compare_text_with_gpt4(text1, text2)
                print(comparison_result)
                # comparison_result = response.choices[0].text
                st.write(comparison_result)
            else:
                st.warning("Please upload both PDF files to compare.")

    if selected=="PDF QnA":
        with st.sidebar:
            with st.expander('advanced'):
                ui_show_debug()
                b_clear()
                ui_model()
                ui_fragments()
                ui_fix_text()
                ui_hyde()
                ui_hyde_summary()
                ui_temperature()
                b_reload()
                ui_task_template()
                ui_task()
                ui_hyde_prompt()

        ui_api_key()
        ui_pdf_file()
        ui_question()
        ui_hyde_answer()
        b_ask()
        ui_output()
        ui_debug()
