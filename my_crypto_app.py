# Använd 'krypto_skapa_modell för att uppdater modellerna till senaste datum

# from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
# import cufflinks as cf

import numpy as np  
import pandas as pd
# from datetime import datetime as dt
from datetime import timedelta
# import pandas_datareader.data as web
import yfinance as yf

from matplotlib import pyplot as plt

import streamlit as st
# from IPython.display import clear_output
from catboost import CatBoostRegressor, Pool, utils
# import time
plt.style.use('fivethirtyeight')

st.set_page_config(
    page_title="JP Crypto Currencies",
    page_icon="🧊",
    # layout="wide",
    initial_sidebar_state="auto"
)

def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )
_max_width_()    

def proc_change(df, ny, gammal, dagar):
    df[ny+'_'+str(dagar)] = df[gammal].pct_change(dagar)

def relret(df):
    rel=df.pct_change()
    cumret = (1+rel).cumprod() - 1
    cumret=cumret.fillna(0)
    return cumret
    
# create vol_x and ret_x columns
def vol_ret_kolumner(df):
    for i in range(10):
        proc_change(df,'ret','Adj Close',i+1)
        proc_change(df,'vol','Volume',i+1)
    return df  
    
def define_prognos_columns(df):
    df = vol_ret_kolumner(df)
    
    df['ret_y'] = df.ret_1.copy()
    df['vol_y'] = df.vol_1.copy()
    
    df.drop(['ret_1','vol_1'],axis=1,inplace=True)
    
    df['year']=pd.DatetimeIndex(df.index).year
    df['month']=pd.DatetimeIndex(df.index).month
    df['wday']=pd.DatetimeIndex(df.index).weekday
    df['day']=pd.DatetimeIndex(df.index).day
    
    return df

def add_row(df):
    # print('shape\t',df.shape)
    last_date = df.iloc[-1:].index
    # print('last_date\t',last_date)
    new_date = last_date + timedelta(days=1)
    # print('NEW DATE\t',new_date)
    # print('NEW DATE[0]\t',new_date[0])
    new_data = pd.DataFrame(df.iloc[-1:], index=[new_date[0]], columns=df.columns)
    
    new_data['year']=None
    new_data['month']=None
    new_data['wday']= None # pd.DatetimeIndex(new_date).weekday
    new_data['day']=  None # pd.DatetimeIndex(new_date).day
    df = pd.concat([df,new_data])
    
    df = define_prognos_columns(df)
    return df

# för predict och graf
def set_kolumner():
    rkolumner = ['vol_y', 'ret_2', 'vol_2', 'ret_3', 'vol_3', 'ret_4', 'vol_4', 'ret_5', 'vol_5', 
                'ret_6', 'vol_6', 'ret_7', 'vol_7', 'ret_8', 'vol_8', 'ret_9', 'vol_9', 'ret_10', 
                'vol_10', 'year', 'month', 'wday', 'day']
    vkolumner = rkolumner[1:]
    
    return vkolumner, rkolumner

# translate the predicted ret_y and vol_y back to Adj Close and Volume scale
# translate the predicted ret_y and vol_y back to Adj Close and Volume scale
def transl_ret_y(predicted_ret_y, predicted_vol_y, previous_AdjClose, previous_Volume):
    predicted_AdjClose = (1+predicted_ret_y) * previous_AdjClose
    predicted_Volume = (1+predicted_vol_y) * previous_Volume
    return predicted_AdjClose, predicted_Volume.astype('int64')
 
def predict_alt_n_days(df,kryptotext, relativ=False):
    vmodel = CatBoostRegressor().load_model(kryptotext+'_vmodel')
    model  = CatBoostRegressor().load_model(kryptotext+'_model')
    
    vkolumner, rkolumner = set_kolumner()
    
    for i in range(5):
        df = add_row(df.copy())
        
        l=df.iloc[-1:].index
        vol_y = vmodel.predict(df.loc[l,vkolumner])[0]
        
        df.loc[l,'vol_y'] = vol_y
        ret_y = model.predict(df.loc[l,rkolumner])[0]
        df.loc[l,'ret_y'] = ret_y
        if not relativ:
            # vi vill ha de Adj Close och Volume omräknade till 'verkliga' värden
            predicted_AdjClose, predicted_Volume = transl_ret_y(ret_y, vol_y, df.iloc[-2]['Adj Close'],df.iloc[-2]['Volume'])
            df.loc[l,'Adj Close'] = predicted_AdjClose
            df.loc[l,'Volume'] = predicted_Volume
        
            df.Volume = df.Volume.astype('int64')
    return df    

def predict_n_days(df,kryptotext, relativ=False):
    
    vkolumner, rkolumner = set_kolumner()
    l=df.iloc[-1:].index
    for i in range(5):
        df = add_row(df.copy())
        nyrad = df.iloc[-1:].index
        vmodel = CatBoostRegressor().load_model(f'{kryptotext}{i+1}_vmodel')
        model  = CatBoostRegressor().load_model(f'{kryptotext}{i+1}_model')
        
        vol_y = vmodel.predict(df.loc[l,vkolumner])[0]
        ret_y = model.predict(df.loc[l,rkolumner])[0]
        df.loc[nyrad,'vol_y'] = vol_y
        df.loc[nyrad,'ret_y'] = ret_y
        if not relativ:
            # vi vill ha de Adj Close och Volume omräknade till 'verkliga' värden
            predicted_AdjClose, predicted_Volume = transl_ret_y(ret_y, vol_y, df.iloc[-2]['Adj Close'],df.iloc[-2]['Volume'])
            df.loc[nyrad,'Adj Close'] = predicted_AdjClose
            df.loc[nyrad,'Volume'] = predicted_Volume
        
            df.Volume = df.Volume.astype('int64')   
    return df    


def add_bollinger(df):
    df['SMA'] = df['Adj Close'].rolling(window=20).mean()
    df['stddev'] = df['Adj Close'].rolling(window=20).std()
    df['Upper'] = df.SMA + 2*df.stddev
    df['Lower'] = df.SMA - 2*df.stddev

    df['Buy_Signal'] = np.where(
        df.Lower > df['Adj Close'], True, False)
    df['Sell_Signal'] = np.where(
        df.Upper < df['Adj Close'], True, False)
    return df


def add_Ichimoku_to_df(df_):
    df = df_.copy()
    # Conversion
    hi_val = df['High'].rolling(window=9).max()
    low_val = df['Low'].rolling(window=9).min()
    df['Conversion'] = (hi_val + low_val) / 2

    # Baseline
    hi_val2 = df['High'].rolling(window=26).max()
    low_val2 = df['Low'].rolling(window=26).min()
    df['Baseline'] = (hi_val2 + low_val2) / 2

    # Span A
    df['SpanA'] = ((df['Conversion'] + df['Baseline']) / 2).shift(26)

    # Span B
    hi_val3 = df['High'].rolling(window=52).max()
    low_val3 = df['Low'].rolling(window=52).min()
    df['SpanB'] = ((hi_val3 + low_val3) / 2).shift(26)

    # Lagging Span
    df['Lagging'] = df['Close'].shift(-26)

    return df


def get_fill_color(label):
    if label >= 1:
        return 'rgba(0,250,0,0.4)'
    else:
        return 'rgba(250,0,0,0.4)'
def plot_ichimoku(df, title):
    # set candlestick green and red
    candle = go.Candlestick(x=df.index, open=df['Open'],
                            high=df['High'], low=df["Low"], close=df['Close'], name="Candlestick") 
    
    df1 = df.copy()
    fig = go.Figure()
    df['label'] = np.where(df['SpanA'] > df['SpanB'], 1, 0)
    df['group'] = df['label'].ne(df['label'].shift()).cumsum()

    df = df.groupby('group')

    dfs = []
    for name, data in df:
        dfs.append(data)

    for df in dfs:
        fig.add_traces(go.Scatter(x=df.index, y=df.SpanA,
                                  line=dict(color='rgba(0,0,0,0)')))

        fig.add_traces(go.Scatter(x=df.index, y=df.SpanB,
                                  line=dict(color='rgba(0,0,0,0)'),
                                  fill='tonexty',
                                  fillcolor=get_fill_color(df['label'].iloc[0])))

    baseline = go.Scatter(x=df1.index, y=df1['Baseline'],
                          line=dict(color='orange', width=2), name="Baseline")

    conversion = go.Scatter(x=df1.index, y=df1['Conversion'],
                            line=dict(color='black', width=1), name="Conversion")

    lagging = go.Scatter(x=df1.index, y=df1['Lagging'],
                         line=dict(color='purple', width=2), name="Lagging")

    span_a = go.Scatter(x=df1.index, y=df1['SpanA'],
                        line=dict(color='green', width=2, dash='dot'), name="Span A")

    span_b = go.Scatter(x=df1.index, y=df1['SpanB'],
                        line=dict(color='red', width=1, dash='dot'), name="Span B")

    fig.add_trace(candle)
    fig.add_trace(baseline)
    fig.add_trace(conversion)
    fig.add_trace(lagging)
    fig.add_trace(span_a)
    fig.add_trace(span_b)

    # set title
    my_title = f'{title}<br><sup><i>zoom in/out i grafen under</i></sup>'
    
    fig.update_layout(title=my_title)
    
    fig.update_layout(height=800, width=1300, showlegend=True , 
                    #   title=go.layout.Title(
                    #       text="Plot Title <br><sup>Plot Subtitle</sup>",
                    #       xref="paper",
                    #       x=0
                    #   ),
                      paper_bgcolor='rgba(255,255,255,0)', plot_bgcolor='rgba(255,255,255,0.8)')

    st.write(fig)

load = st.container()
graf = st.container()

alternativ_text = """
                Varje dags prognos görs individuellt utan påverkan på varandra.    
                
                Alternativ (rekommenderas ej): Varje dags prognos bygger på föregåend dags prognos (stegvis beräknas prognos för dag 1 som används som input till dag 2 osv). Denna version kan ibland överdriva en svag trend.    
                """
global valuta
valuta='ETH-USD'
typ=1
typtxt = st.sidebar.selectbox('Vad göra?',('Graf per kryptovaluta','5-dagars relativ prognos för alla krypto','Jämföra krypto med Sthlm-börsen'),index=0)
if typtxt[:3] == 'Jäm':             # jämför med OMX
    typ=1
elif  typtxt[:3] == '5-d':          # 5-dagars prognos för alla
    typ=3
else:                               # detaljerad graf (ev prognos) per valuta
    typ=2
    
if typ==2:    # graf per valuta
    # graftyp = 'Enkel graf med prognos'
    graftyp = st.sidebar.selectbox('välj graf-typ', ('Enkel graf med prognos', 'Bollinger-graf med prognos', 'Ichimoku-graf'))

    kryptotext = st.sidebar.selectbox('vilken valuta',('ETH (Ethereum)','BTC (Bitcoin)',
                                                    'BCH (Bitcoin Cach)','ZRX (0x)','XRP'),index=0)
    kryptotext = kryptotext[:3]
    if kryptotext=='ETH':
        valuta = 'ETH-USD'
    elif kryptotext=='BTC':
        valuta = 'BTC-USD'   
    elif kryptotext=='BCH':
        valuta = 'BCH-USD'   
    elif kryptotext=='ZRX':
        valuta = 'ZRX-USD'   
    elif kryptotext=='XRP':
        valuta = 'XRP-USD'  
            
    with load:
        # valuta = web.DataReader(valuta,'yahoo') # Etherium
        valuta = yf.download(valuta,progress=False)

        # st.write('de 5 sista dagarna exrtrakt',valuta.iloc[-5:][['Adj Close', 'Volume',]])
            
    # st.write('lastval',valuta.iloc[-1:].index[0])
      
    tidsram = '30 dagar'   
    with graf:  
        if graftyp[:8] != 'Ichimoku':
            tidsram = st.sidebar.selectbox('tidsram för graf', ('15 dagar', '30 dagar', '90 dagar', 'från inköp'), 1)
            if tidsram == '15 dagar':
                tidsram = 15
            elif tidsram == '30 dagar':
                tidsram = 30
            elif tidsram == '90 dagar':
                tidsram = 90
            elif tidsram == 'från inköp':
                startdat = '2021-04-12'
                partlen = len(valuta.loc[:startdat])-1
                totlen = len(valuta)
                tidsram = totlen-partlen
            else:
                pass

            data = define_prognos_columns(valuta.copy())

            alternativ = st.sidebar.selectbox(
                'Alternativ prognos', ('Ja', 'Nej'), index=1)
            if alternativ == 'Nej':
                data = predict_n_days(data.copy(), kryptotext)
            else:
                data = predict_alt_n_days(data.copy(), kryptotext)

            with st.sidebar.expander('Förklaring av alternativ prognos'):
                st.info(alternativ_text)

            # st.write('tidsram =',tidsram,'dagar för grafen')
            lastdate=valuta.iloc[-1:].index
            st.write('Senast kända datum', str(lastdate[0])[:10]+'. (Efter den röda prickade linjen följer en 5 dagars prognos)')
            
            ### plot Adj Close ###
            fig = plt.figure(figsize=(16,6))
            ax = fig.add_subplot(1,1,1)
            
            if alternativ=='Ja':
                ax.set_title(kryptotext+' "Adjusted Close" Alternativ prognos')
            else:
                ax.set_title(kryptotext+' "Adjusted Close"')   
    
            if graftyp[:4]=='Boll':
                df = add_bollinger(data)
                # ax.plot(
                #     df.index,
                #     df["Adj Close"], color='b',
                # )
                df = df.iloc[-tidsram:]
                maxa = df['Upper'].max()
                mina = df['Lower'].min()

                ax.plot(
                    df.index,
                    df[["Adj Close",'SMA','Upper','Lower']]
                )    
                ax.fill_between(df.index,df.Upper,df.Lower,color='grey',alpha=0.3)
                ax.legend(['Pris (Adj Close)','Simple Moving Avg','Övre','Undre'])

                # ax.set_xlabel("Datum")
                ax.set_ylabel("$USD")
                ax.tick_params(axis='x', rotation=66)
                    
                ax.vlines(lastdate,mina,maxa,colors='r', linestyles='dotted')
                st.write(fig)
                    
                fig2 = plt.figure(figsize=(18, 4))

                ax2 = fig2.add_subplot(1,1,1)

                ax2.set_title(kryptotext+' Volymer')
                
                ax2.set_ylabel("Miljoner")
                ax2.tick_params(axis='x',rotation=66)
                ax2.plot(
                    df.index,
                    df['Volume']/1000000,color='g'
                )
                maxv = df['Volume'].max()/1000000
                minv = df['Volume'].min()/1000000
                ax2.vlines(lastdate,minv,maxv,colors='r', linestyles='dotted')
                
                st.write(fig2)
            else:
                # Enkel graf med prognos
                df = data.iloc[-tidsram:]
                maxa = df['Adj Close'].max()
                mina = df['Adj Close'].min()

                ax.set_ylabel("$USD")
                ax.tick_params(axis='x', rotation=66)
                ax.vlines(lastdate, mina, maxa,
                          colors='r', linestyles='dotted')
                ax.plot(
                    df.index,
                    df[["Adj Close"]]
                )
                st.write(fig)
                fig2 = plt.figure(figsize=(18, 4))

                ax2 = fig2.add_subplot(1, 1, 1)

                ax2.set_title(kryptotext+' Volymer')

                ax2.set_ylabel("Miljoner")
                ax2.tick_params(axis='x', rotation=66)
                ax2.plot(
                    df.index,
                    df['Volume']/1000000, color='g'
                )
                maxv = df['Volume'].max()/1000000
                minv = df['Volume'].min()/1000000
                ax2.vlines(lastdate, minv, maxv,
                           colors='r', linestyles='dotted')

                st.write(fig2)
        else: # Ichimoku
            startdat = '2021-04-12'
            partlen = len(valuta.loc[:startdat])-1
            totlen = len(valuta)
            totram = totlen-partlen
            dfi = valuta.iloc[-totram:]
            # background white
            dfi = add_Ichimoku_to_df(dfi)
            plot_ichimoku(dfi,f'{kryptotext} "Adjusted Close"')
            
        if st.button('inspektera '+kryptotext+' data'):
            st.write('tidsram', tidsram)
            st.dataframe(data.iloc[-tidsram:])
        else:
            pass    
        
elif typ==1:   # Jämför med OMX
    tickers = ['BTC-USD','BCH-USD','ETH-USD','XRP-USD','ZRX-USD']
    
    with st.spinner('ta det lugnt'):
        # df = relret(web.DataReader(tickers,'yahoo',start)['Adj Close']) # alla mina krypto
        start='2021-04-13'
        df = relret(yf.download(tickers,start=start,progress=False)['Adj Close'])
        oldestdate = str(df.index[0])[:10] 
         
        title='Relativ utv av mina kryptovalutor och OMX30'
        # omx = relret(web.DataReader(['^OMX'],'yahoo',start)['Adj Close']) # Stockholm 30 index
        omx = relret(yf.download(['^OMX'],start=start,progress=False)['Adj Close'])
        df=pd.merge(df,omx,left_index=True,right_index=True,how='outer')
        df.rename(columns = {'BTC-USD':'Bitcoin','BCH-USD':'Bitcoin Cash','ETH-USD':'Ethereum','XRP-USD':'XRP','ZRX-USD':'0x (ZRX)','Adj Close':'OMX30'},inplace=True)
       
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(1,1,1)
    # df=data.iloc[-tidsram:]
    
    ax.set_title(title,size=24,color='b')
    ax.plot(
        df.index,
        df,
        linewidth=2
    )   
    
    ax.legend(df.columns,handletextpad=1, fontsize = 12.0,)
    
    ax.set_ylabel("relativ utveckling",fontsize = 18.0)
    ax.tick_params(axis='x', rotation=45, labelsize=12.0)
    st.write(fig)
    
    with st.sidebar.expander('Förklaring'):
        st.info("""Grafen visar utveckling av mina kryptovalutor och OMX30 relativt varandra   
            OMX30 är ett snitt av Stockholmsbörsens 30 mest omsatta aktier.   
            Allt startar från min inköpsdatum av kryptovalutor """ + oldestdate +      
            """  \nAtt OMX30-linjen har tomrum beror på helgdagar då börsen är stängd""")
        
else: # typ==3 prognos för alla
    tickers = ['BTC-USD','BCH-USD','ETH-USD','XRP-USD','ZRX-USD']
    alternativ = st.sidebar.selectbox('Alternativ prognos',('Ja','Nej'),index=1)
    with st.sidebar.expander('Förklaring av alternativ prognos'):
        st.info(alternativ_text)
    df=pd.DataFrame()
    with st.spinner('ta det lugnt'):
        # df = relret(web.DataReader(tickers,'yahoo',start)['Adj Close']) # alla mina krypto
        start='2021-08-01'
        allt = yf.download(tickers,start=start,progress=False)[['Adj Close','Volume','Close']]
        df = relret(allt)
        
        today = str(df.index[-1])[:10] 
         
        title='5 dagars prognos av mina kryptovalutor'
        
        df_prognos = pd.DataFrame()
        for ticker in tickers:   
            the_text = ticker[:3]
            data = allt[[('Adj Close',ticker),('Volume',ticker),('Close',ticker)]].copy()
            data = data.dropna()
            data.columns = ['Adj Close', 'Volume','Close']
            data = define_prognos_columns(data)  # remove unused columnes and add new columns
            cols = list(data.columns)
            if alternativ=='Nej':
                # st.write(data)
                data = predict_n_days(data.copy(),the_text)
            else:
                data = predict_alt_n_days(data.copy(),the_text)
            
            df_prognos[the_text] = relret(data.iloc[-6:]['Adj Close'] )
            
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(1,1,1)
    
    ax.set_title(title,size=30,color='b')
    ax.plot(
        df_prognos.index,
        df_prognos,
        linewidth=2
    )   
    
    ax.legend(df_prognos.columns,handletextpad=1, fontsize = 12.0)
    
    ax.set_ylabel("relativ prognos",fontsize = 18.0)
    ax.tick_params(axis='x', rotation=45, labelsize=12.0)
    st.write(fig)
    
    with st.sidebar.expander('Förklaring graf'):
        st.info("""Grafen visar prognos för mina kryptovalutor relativt varandra.   
            Alla startar från dagens datum """ + today + ' med värdet 0 och därefter prognos av relativ utveckling')
    
