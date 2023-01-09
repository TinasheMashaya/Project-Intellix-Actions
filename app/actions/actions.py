import wikipedia
import sympy
from typing import Any, Text, Dict, List
from sympy.printing import latex as spl
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import UserUtteranceReverted
from rasa_sdk.events import SlotSet
from sympy import cos ,sin ,tan,  trigsimp , Integral , series ,Derivative ,solve_poly_system ,sympify,integrate ,diff ,solve ,expand ,factor ,expand_trig
from sympy.abc import*
import os
import io
import openai
from google.cloud import storage
from sympy.parsing.sympy_parser import (parse_expr,standard_transformations, implicit_multiplication_application)
from sympy.plotting import plot
from chemica import Chemica
import requests
from mendeleev import element
import chemistry_tools.pubchem.description as ds
import chemistry_tools.formulae.compound as nc
from webxplore import WebScraper
transformations = standard_transformations + (implicit_multiplication_application,)
import datetime as dt
# import mysql.connector
import psycopg2
import json

openai.api_key = "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"
model_engine = "text-davinci-003"
# mydb = mysql.connector.connect(
#   host="localhost",
#   user="qubit_user",
#   password="qubit_2022",
#   database = "schoolbooks"
 
# )
# con = psycopg2.connect(database="books", user="qubit_space",password="qubit_2022", host="127.0.0.1", port="5432")
# print("Database opened successfully")
# cur = con.cursor()

# mycursor = mydb.cursor()
def simplify(exp):
    remove_power = exp.replace('^', '**')
    return parse_expr(str(remove_power), transformations=transformations)



def copy_sort(final_eqn ) :
    if isinstance(final_eqn, list)== True:
        print(isinstance(final_eqn, list))
        print('This is a list')
        sorted_copy = [ ]
        chem_string= ''
        i = 0
        
        while  i<=0:
            minimum = ''
            
            for element in range( 0 , len( final_eqn) ) :
                
                i = +1
                minimum = final_eqn[element].result
                #print(i)
                #print( '\tInserting' , minimum)
                sorted_copy.append(minimum)
                chem_string += '\\n'  + minimum + '\\n' 
                chemistry_string = ''.join(chem_string.split())[:-2]
           
        return chemistry_string
    else:
        print('This is not a list')
        return final_eqn.result

def splitts(lhs, expr,terms_per_line=3):
    """
    This function is written to handle displaying long latex equations.
    """
    latex = r'\begin{aligned}' + '\n'
    terms = expr.as_ordered_terms()
    n_terms = len(terms)
    term_count = 1
    for i in range(n_terms):
        term = terms[i]
        term_start = r''
        term_end = r''
        sign = r'+'
        if term_count > terms_per_line:
            term_start = '&'
            term_count = 1
        if term_count == terms_per_line:
            term_end = r'\ldots \\' + '\n'
        if term.as_ordered_factors()[0]==-1:
            term = -1*term
            sign = r'-'
            
        if i == 0: 
            
            if sign == r'+': sign = r""
            latex += r'{:s} =& {:s} {:s} {:s} {:s}'.format(sympy.latex(lhs),
                                                     term_start,sign,sympy.latex(term),term_end)

        elif i == n_terms-1: # end
            latex += r'{:s} {:s} {:s} {:s}'.format(term_start,sign,sympy.latex(term),term_end)
        else: # middle
            latex += r'{:s} {:s} {:s} {:s}'.format(term_start,sign,sympy.latex(term),term_end)
        term_count += 1
    latex += r'\end{aligned}' + '\n'
    #return Latex(latex)
    #print(Latex(latex))
    #print(term)
    #print(latex)
    #print(spl(latex))
    return spl(latex)


class userRequestBook(Action):
    
    def name(self) -> Text:
        return "action_user_request_book"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

    

        book_request = str(tracker.get_slot("book_info"))
        print('EXTRACTED_VALUE',book_request)

        # sql =  "SELECT description,imageUrl FROM libooks  WHERE MATCH (description) AGAINST ('{}' IN NATURAL LANGUAGE MODE) LIMIT 1".format(book_request)
        # SELECT Title FROM books  WHERE MATCH (Title,Author,Genre,Publisher) AGAINST ('Python' IN NATURAL LANGUAGE MODE) LIMIT 1
        # mycursor.execute(sql)
        cur.execute("SELECT websearch_to_tsquery('english','{}')".format(book_request))
        

        rows = cur.fetchall()
        s = rows[0][0]
        f = s.replace("'", "")
        g = f.replace("&","|")

        z = "'{}'".format(g)
        print(z)
        cur.execute("SELECT description ,imageurl,ts_rank(to_tsvector('english',description), to_tsquery('english', {})) AS rank FROM libooks ORDER BY rank DESC LIMIT 5".format(z))

        queryResult = cur.fetchall()
        dateString = '{}:{}'.format(dt.datetime.now().hour, dt.datetime.now().minute)
        fulfillmentText = {
            'description':queryResult[0][0],
            'imageUrl': queryResult[0][1],
            'timestamp': dateString,
            'request_type':'db.books.request',
            'actions':'actions'
            
        }
        print("Operation done successfully")
        # con.close()
        dispatcher.utter_message(json_message =fulfillmentText )
    

        return [SlotSet("book_info", None)]

class matheSquareRoots(Action):

    def name(self) -> Text:
        return "action_math_square_roots"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        square_root = 0

        square_root = int(tracker.get_slot("math_square_roots"))
        #print('here num1 = {0}'.format(square_root))
        
        square_root_result = sympy.sqrt(square_root)
        lhs =  'Answer'
        square_root_answer = splitts(lhs,square_root_result)
        
        fulfillmentText =  str(square_root_answer)
        #fulfillmentText = str(square_root_answer)
        dispatcher.utter_message(text=fulfillmentText)
        return [SlotSet("math_square_roots", None)]

class EntityPerfomance(Action):
    
    def name(self) -> Text:
        return "action_perfomance"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

       

        fulfillmentText = str(tracker.get_slot("perfomance"))
        
        dispatcher.utter_message(text=fulfillmentText)
        return []

class FallBackAction(Action):
    
    def name(self) -> Text:
        return "action_default_fallback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        response = ""
        try:
            user_input = str(tracker.latest_message.get('text'))
            print(user_input)
            completion = openai.Completion.create(
                engine=model_engine,
                prompt=user_input,
                max_tokens=1024,
                temperature=0.5,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            # url = "https://api.writesonic.com/v2/business/content/chatsonic?engine=premium"
            # headers = {
            #     "accept": "application/json",
            #     "content-type": "application/json",
            #     "X-API-KEY": "e9164a47-9ae9-4ecc-8115-97e359a6fc9a"
            # }
            # payload = {
            #         "enable_google_results": "true",
            #         "enable_memory": False,
            #         "input_text":user_input
            # }
            # response = requests.post(url, json=payload, headers=headers)
            response = completion.choices[0].text
            print(response)
            # response = json.loads(response.text)
            # print(response)
            
            response= response["message"]
        except Exception as e:
            response = str(e)
            pass

        print(response)
        dispatcher.utter_message(text=response)
        return []

class wikiSearch(Action):
    
    def name(self) -> Text:
        return "action_wiki_search"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        summary = str(tracker.get_slot("search"))
        search_list = wikipedia.search(summary)
        disamb_helper = search_list[:4]
        unwanted_chars = ['(',')']
        for i in unwanted_chars:
            new_helper = [s.replace("(", "") for s in  disamb_helper]
            disamb_list = [s.replace(")", "") for s in   new_helper]

        print(disamb_list)
        try:
            
            summary_result = wikipedia.summary(disamb_list[0])
            print('Section 1:', print(disamb_list[0]))
         
            print('Value being removed from section 1:',disamb_list[0])
            disamb_list.pop(0)
            suggestion = "\nYou may be interested in any of the following.Try :\n\nsearch {}\nsearch {}\nsearch {}".format(disamb_list[0],disamb_list[1],disamb_list[2])
            
        except (wikipedia.exceptions.DisambiguationError,wikipedia.exceptions.PageError, wikipedia.exceptions.RedirectError) as e:
            print(e)
            print('Section 2:',disamb_list[1])
            summary_result= wikipedia.summary(disamb_list[1])
            print('Value being removed from section 2:',disamb_list[1])
            disamb_list.pop(1)
            print('Value being removed from section 2:',disamb_list[0])
            disamb_list.pop(0)
            suggestion = "Related:\n\nsearch {}\nsearch {}".format(disamb_list[0],disamb_list[1])
            
        print(disamb_list)

        fulfillmentText = summary_result+"\n{}".format(suggestion)
        dispatcher.utter_message(text=fulfillmentText)

        return [SlotSet("search", None)]

class mathIntegration(Action):
    
    def name(self) -> Text:
        return "action_math_integration"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        print('Source of error:',str(tracker.latest_message['entities']))
        
        trans_input = str(tracker.get_slot("math_integration"))
      
        print('INPUT:',trans_input)
        print(type(trans_input))
        integral = simplify(trans_input)
        print(integral)

    
        try:

            expr = sympify(integrate(integral, x))
            lhs = Integral(integral,x)
            integral_result = splitts(lhs,expr)
            print(expr)
            print(type(integral_result))
            print(integral_result)
            fulfillmentText =  integral_result
            dispatcher.utter_message(text=fulfillmentText)

            return [SlotSet("math_integration", None)]

        except(AttributeError):
            print('Source of error:',integral)

class mathDifferentiation(Action):
    
    def name(self) -> Text:
        return "action_math_differentiation"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

       
        x,y = symbols('x y')
        diff_q = x**2 + x + 1
        diff_input = str(tracker.get_slot("math_differentiation"))
       
        diff_q = simplify(diff_input)
        
        diff_result = sympify(diff_q)
       
        diff_result_answer = diff(diff_result, x)
        lhs = Derivative(diff_result)
        diff_final  = splitts(lhs, diff_result_answer)
        
        fulfillmentText =  diff_final
        dispatcher.utter_message(text=fulfillmentText)


        return [SlotSet("math_differentiation", None)]

class mathSeries(Action):
    
    def name(self) -> Text:
        return "action_math_series"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

       
        x,y = symbols('x y')
        diff_q = x**2 + x + 1
        series_q  = simplify(str(tracker.get_slot("math_series")))

        series_result = sympify(series_q)
       
        series_result_answer = series(series_result, x)
        lhs = series_result
        series_final = splitts(lhs,series_result_answer)
        
        fulfillmentText =  series_final
        dispatcher.utter_message(text=fulfillmentText)


        return [SlotSet("math_series", None)]

class mathSimplifyPolynomial(Action):
    
    def name(self) -> Text:
        return "action_math_simplify_polynomials"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

       
        x,y = symbols('x y')
        poly_q  = sympify(str(tracker.get_slot("math_simplify_polynomials")))
       
        
        poly_result = sympify(poly_q)
        lhs =  poly_result 
        poly_result_ = solve(poly_result, x)
        poly_result_answer = splitts(lhs, poly_result_)
        
        fulfillmentText =  str(poly_result_answer)
        dispatcher.utter_message(text=fulfillmentText)


        return [SlotSet("math_simplify_polynomials", None)]

class mathExpandPolynomial(Action):
    
    def name(self) -> Text:
        return "action_math_expand_polynomials"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

       
        x,y = symbols('x y')
        poly_expand_q  = sympify(str(tracker.get_slot("math_expand_polynomials")))  
        lhs = poly_expand_q
        poly_expand = expand( poly_expand_q, x)
        poly_expand_result = splitts(lhs,poly_expand)

        fulfillmentText =  poly_expand_result
        dispatcher.utter_message(text=fulfillmentText)


        return [SlotSet("math_expand_polynomials", None)]

class mathUnivariateEquations(Action):
    
    def name(self) -> Text:
        return "action_math_univariate_equations"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

       
        x,y = symbols('x y')
        u_eqn  = sympify(str(tracker.get_slot("math_univariate_equations")))  
       
        #poly_expand_q = sympify(query_result.get('parameters').get('univariate_equation_var'))
        univariate_equation = solve(u_eqn)
        lhs = 'solution(s)'
        univariate_equation_result = splitts(lhs, univariate_equation)      
        fulfillmentText =  univariate_equation_result 
        #fulfillmentText = str(integral_result)
        
        dispatcher.utter_message(text=fulfillmentText)


        return [SlotSet("expression", None)]

class mathMultiVariateEquations(Action):
    
    def name(self) -> Text:
        return "action_math_multivariate_equations"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

       
        x,y = symbols('x y')
        m_eqn1  = sympify(str(tracker.get_slot("math_multivariate_equations"))) 
        m_eqn2  = sympify(str(tracker.get_slot("math_multivariate_equations_2")))  
       
        m_result = solve_poly_system([m_eqn1,m_eqn2])
        lhs = 'solution(s)'
        final_result = splitts(lhs,m_result)
        fulfillmentText =  final_result 
        dispatcher.utter_message(text=fulfillmentText)


        return [SlotSet("math_multivariate_equations", None),SlotSet("math_multivariate_equations_2", None)]

class mathFactorise(Action):
    
    def name(self) -> Text:
        return "action_math_factorise"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

       
        x,y = symbols('x y')
        
        poly_factorise_q  = sympify(str(tracker.get_slot("math_factorise"))) 
       
        poly_factorise = factor(poly_factorise_q,x)
        lhs  =  poly_factorise_q
        poly_factorise_result = splitts(lhs, poly_factorise)
        
        fulfillmentText = poly_factorise_result
        dispatcher.utter_message(text=fulfillmentText)


        return [SlotSet("expression", None)]

class mathTrigExpansion(Action):
    
    def name(self) -> Text:
        return "action_math_trig_expansion"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

       
        x,y = symbols('x y')  
        expandtrig   = sympify(str(tracker.get_slot("math_trig_expansion"))) 
        trig_expand = expand_trig(expandtrig) 
        lhs = sympify(expandtrig)
        trig_result_expand = splitts(lhs,trig_expand)
            
        fulfillmentText =  trig_result_expand
        dispatcher.utter_message(text=fulfillmentText)
        return [SlotSet("math_trig_expansion", None)]

class mathTrigSimplifications(Action):
    
    def name(self) -> Text:
        return "action_math_trig_simplifications"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

       
        x,y = symbols('x y')  
        simptrig   = sympify(str(tracker.get_slot("math_trig_simplifications")))
        trig_simp = trigsimp(simptrig) 
        lhs = sympify(simptrig)
       
        trig_result = splitts(lhs,trig_simp)
        fulfillmentText = trig_result
        dispatcher.utter_message(text=fulfillmentText)

        return [SlotSet("math_trig_simplifications", None)]

class mathStandardForm(Action):
    
    def name(self) -> Text:
        return "action_math_standard_form"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

       
        x,y = symbols('x y')  
        standard_form = sympify(int(tracker.get_slot("math_standard_form"))) 
       
        standard_form_result = sympy.float(standard_form)

        
        fulfillmentText = str( standard_form_result)
        dispatcher.utter_message(text=fulfillmentText)


        return [SlotSet("math_standard_form", None)]

class mathSketch(Action):
    
    def name(self) -> Text:
        return "action_math_sketch"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

       
        x,y = symbols('x y')  
        
       
        sketch_input = sympify(str(tracker.get_slot("math_sketch")))
        sketch = plot(sketch_input)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'intellixbot-jqcj-9f56aa4052bf.json'
        client = storage.Client()
        bucket = client.get_bucket('intellixbot-jqcj.appspot.com')
        buf = io.BytesIO()
        sketch.save(buf)
        
        blob = bucket.blob(sketch_input)
        blob.upload_from_string(buf.getvalue(),content_type='image/png')
        buf.close()
        url = blob.public_url
        
        fulfillmentText = url   


        return [SlotSet("math_sketch", None)]

class chemistryEquations(Action):
    
    def name(self) -> Text:
        return "action_chemistry_equations"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

       
        para_eqn1 = str(tracker.get_slot("chemistry_equations"))
        para_eqn2 = str(tracker.get_slot("chemistry_equations_2"))
        eqn = (Chemica.solve(para_eqn1, para_eqn2))
        final_eqn = copy_sort(eqn)
        full_equation = str(final_eqn)
        url = 'https://chemequations.com/en/?s={}+%2B+{}&ref=input'.format(para_eqn1,para_eqn2)
        webScraper = WebScraper.ScrapeWebsite(url)
        equation_type = webScraper.return_article()
        if not full_equation: 
            fulfillmentText = 'Ooops !! I could not solve your chemical equation.Please check for errors and send it again'
        else:
            string_equation = str(full_equation)
            fulfillmentText = 'Existing Equation(s) : \\n{}\n\n{}'.format(string_equation, equation_type)
        dispatcher.utter_message(text=fulfillmentText)


        return [SlotSet("chemistry_equations", None),SlotSet("chemistry_equations_2", None)]

class chemistryElements(Action):
    
    def name(self) -> Text:
        return "action_chemistry_elements"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

       
       
        chemical_info = str(tracker.get_slot("chemistry_elements"))
        ci = element(chemical_info)
        descriptions = ci.description
        use = ci.uses
        names = ci.name
        location = ci.discovery_location
        discover = ci.discoverers
        year = ci.discovery_year
        source = ci.sources
        atomic_num = ci.atomic_number
        mass_num  = ci.mass_number
        boiling_pt = ci.boiling_point
        melting_pt = ci.melting_point
        origin_name = ci.name_origin
        oxidation_states = ci.oxistates
        oxide = ci.oxides()
        electron_config = ci.ec
        group_ = ci.group_id
        period_ = ci.period
        element_result = "{}\n{}\n\nUses\n{}\n\nSources\n{}\n\nDiscovery\nDiscovered in {} by {} in {}\nOrigin:  {}\n\nChemical Properties\nAtomic Number: {}\nMass Number: {}\nBoiling Point: {}\nMelting Point: {}\nOxides: {}\nOxidation States: {}\nGroup: {}\nPeriod: {}\nElectronic Configuration : {}".format(names,descriptions, use,source,location,discover,year,origin_name,atomic_num,mass_num,boiling_pt,melting_pt,oxide,oxidation_states,group_,period_,electron_config)
        fulfillmentText = element_result
        
        dispatcher.utter_message(text=fulfillmentText)


        return [SlotSet("chemistry_elements", None)]

class chemistryCompounds(Action):
    
    def name(self) -> Text:
        return "action_chemistry_compounds"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

       
        chemical_compound = str(tracker.get_slot("chemistry_compounds"))
        wiki_description =''
        pubchem_description=''
        source_pubchem = 'Source Pubchem'
        source_wikipedia = 'Source Wikipedia'
       
        
        try:
            name_check = nc.Compound( chemical_compound).name
            search_description = wikipedia.search( name_check)
        
            wiki_description = wikipedia.summary(search_description[0])
           
        except (ValueError,wikipedia.exceptions.PageError, wikipedia.exceptions.RedirectError,wikipedia.exceptions.DisambiguationError,wikipedia.exceptions.HTTPTimeoutError,wikipedia.exceptions.WikipediaException) as error:
            print(error)
            try:
                
                wiki_description = wikipedia.summary(chemical_compound)
                print('Second Try')
                print(wiki_description)
            except(ValueError,wikipedia.exceptions.PageError, wikipedia.exceptions.RedirectError,wikipedia.exceptions.DisambiguationError) as error2: 
                print('Second Except')
                print(error2)
                source_wikipedia = ''   
                wiki_description = 'No data obtained from Wikipedia'
      
        try:
             pubchem_description = ds.get_description(chemical_compound)
        except ValueError as k:
            print(k)
            source_pubchem = ''
            pubchem_description = 'No data obtained from Pubchem'
             
        fulfillmentText = "{}\n\n{}\n\n{}\n\n{}" .format(pubchem_description,source_pubchem,wiki_description,source_wikipedia)   
        dispatcher.utter_message(text=fulfillmentText)


        return [SlotSet("chemistry_compounds", None)]

 