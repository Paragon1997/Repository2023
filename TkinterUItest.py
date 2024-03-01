import tkinter as tk
import customtkinter
from CTkMessagebox import CTkMessagebox

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from DEDlib import *
import warnings
warnings.filterwarnings("ignore",category=RuntimeWarning)

from tempfile import TemporaryFile
import pickle

import json
from json import JSONEncoder
import screeninfo

#fDos if N reached in json
#other main functions in seperate windows (graphene, S etc.)

def iterationDED(reset=False):
    global omega,Npoles,c,pbar,eta,Hn,n,AvgSigmadat,Nfin,nd
    while not reset:
        NewM,nonG,_=Startrans(Npoles,np.sort(Lorentzian(omega,DEDargs[5],Npoles,DEDargs[4],DEDargs[3])[1]),omega,eta)
        H0,H=HamiltonianAIM(np.repeat(NewM[0][0],DEDargs[8]),np.tile([NewM[k+1][k+1] for k in range(len(NewM)-1)],(DEDargs[8],1)),np.tile(NewM[0,1:],(DEDargs[8],1)),DEDargs[2],DEDargs[3],DEDargs[9],DEDargs[10],Hn)
        try: (MBGdat,Boltzmann,Ev0),reset=Constraint(DEDargs[6],H0,H,omega,eta,c,n,DEDargs[11],np.array([ar<DEDargs[0] for ar in Nfin]))
        except (np.linalg.LinAlgError,ValueError,scipy.sparse.linalg.ArpackNoConvergence): (MBGdat,Boltzmann,Ev0),reset=(np.zeros(len(omega),dtype='complex_'),np.zeros(len(DEDargs[11])),np.array([])),False
        if np.isnan(1/nonG-1/MBGdat+DEDargs[3]).any() or np.array([i>=1000 for i in np.real(1/nonG-1/MBGdat+DEDargs[3])]).any(): reset=False
    Nfin,AvgSigmadat,nd=Nfin+Boltzmann,AvgSigmadat+(1/nonG-1/MBGdat+DEDargs[3])*Boltzmann[:,None],nd+np.conj(Ev0).T@sum(Hn[0]).data.tocoo()@Ev0*Boltzmann
    #print('{0:.16f}'.format(nd[0]))#np.conj(Ev0).T@sum(Hn[0]).data.tocoo()@Ev0,Boltzmann,np.conj(Ev0).T@sum(Hn[0]).data.tocoo()@Ev0*Boltzmann,
    if DEDargs[6]=='sn': pbar.n+=1
    else: pbar.n=int(min(Nfin))
    pbar.refresh()

def counter():#N,Nmax
    global paused,stopped,started,DEDargs,pbar,Nfin,number,app
    #print(count_var.get())
    if not paused and pbar.n!=DEDargs[0]:
        iterationDED()
        #count_var.set(f"{pbar.n}/{DEDargs[0]}")
        #progressbar.set(pbar.n/DEDargs[0])
        number=pbar.n
        app.progressbar_1.set(pbar.n/DEDargs[0])
    if not stopped and pbar.n!=DEDargs[0]:
        #pbar.refresh()
        if DEDargs[1]>5: 
            if pbar.n%1000==0 and pbar.total>1000:savedata()
            app.after(100,counter)
        else: app.after(1,counter)
    elif pbar.n==DEDargs[0]:
        pbar.close()

    else:
        #stopped=False
        started=False
        number=0
        app.progressbar_1.set(0)
        pbar.n=0
        pbar.reset()

def startDED():
    global started,paused,number,istar,pbar,stopped,app
    istar+=1
    if istar==1:
        if not started: 
            app.start_button.configure(state="disabled")
            started=True
            stopped=False
            pbar.start_t = pbar._time()
            app.N_Entry.configure(state="disabled")
            app.U_Entry.configure(state="disabled")
            app.Sigma_Entry.configure(state="disabled")
            app.Ed_Entry.configure(state="disabled")
            app.Gamma_Entry.configure(state="disabled")
            app.scaling_optionemenu.configure(state="disabled")
            app.sidebar_button_3.configure(state="disabled")
        counter()


def pauseDED():
    global paused,started
    if started:
        paused = not paused 

def stopDED():
    global stopped,started,paused,number,istar,omega,app
    savedata()
    if not stopped and started:
        stopped=True
        paused=False
        number=0
        istar=0

def fileloader():
    global app,started,omega,AvgSigmadat,Nfin,Lor,DEDargs,eta,pbar,Npoles,Hn,n,nd,c,number
    if not started:
        try:
            data=AvgSigmajsonfileR(app.entry.get())
            Nfin,omega,AvgSigmadat,nd=np.array(data["Nfin"]),np.array(data["omega"]),np.array(data["AvgSigmadat"]*data["Nfin"]).squeeze(),np.array(np.array(data["nd"],dtype=np.complex128)*np.array(data["Nfin"],dtype=np.float64),dtype=np.complex128)
            DEDargs=[data["Ntot"],data["poles"],data["U"],data["Sigma"],data["Ed"],data["Gamma"],data["ctype"],data["Edcalc"],data["Nimpurities"],data["U2"],data["J"],data["Tk"],data["etaco"]]
            pbar.total=DEDargs[0]
            pbar.n=data["Nit"]
            number=pbar.n
            app.progressbar_1.set(pbar.n/DEDargs[0])
            Lor=Lorentzian(omega,DEDargs[5],DEDargs[1],DEDargs[4],DEDargs[3])[0]
            Npoles=int(DEDargs[1]/DEDargs[8])
            c,eta=[Jordan_wigner_transform(i,2*DEDargs[1]) for i in range(2*DEDargs[1])],DEDargs[12][0]*abs(omega)+DEDargs[12][1]
            (Hn,n)=Operators(c,DEDargs[8],DEDargs[1])
            pbar.refresh()

            app.U_Entry.delete(0,last_index=tk.END)
            app.U_Entry.insert(0,str(DEDargs[2]))
            app.Sigma_Entry.delete(0,last_index=tk.END)
            app.Sigma_Entry.insert(0,str(DEDargs[3]))
            app.Ed_Entry.delete(0,last_index=tk.END)
            app.Ed_Entry.insert(0,str(DEDargs[4]))
            app.Gamma_Entry.delete(0,last_index=tk.END)
            app.Gamma_Entry.insert(0,str(DEDargs[5]))
            app.scaling_optionemenu.set(str(DEDargs[1]))
            if number>DEDargs[0]: 
                app.N_Entry.delete(0,last_index=tk.END)
                app.N_Entry.insert(0,str(number))
            else:
                app.N_Entry.delete(0,last_index=tk.END)
                app.N_Entry.insert(0,str(DEDargs[0]))                
            app.U_Entry.configure(state="disabled")
            app.Sigma_Entry.configure(state="disabled")
            app.Ed_Entry.configure(state="disabled")
            app.Gamma_Entry.configure(state="disabled")
            app.scaling_optionemenu.configure(state="disabled")

            #print(str(nd),str(data["Nfin"]),str(data["nd"]))#,np.finfo(nd).precision,np.finfo(data["nd"]).precision)
            #print('{0:.16f}'.format(nd[0]),'{0:.16f}'.format(nd[0]/Nfin[0]),'{0:.16f}'.format(data["Nfin"][0]))
        except IOError or FileNotFoundError:
            app.entry.delete(0,last_index=tk.END)
            app.entry.insert(0, 'Try again')

def savfilename():
    global app
    if not app.entry_2.get().endswith(".json"):
        app.entry_2.delete(0,last_index=tk.END)
        app.entry_2.insert(0, 'Try again')

def AvgSigmajsonfileR(name):#dir
    data=json.load(open(name))
    data["AvgSigmadat"]=np.array(data["AvgSigmadat"],dtype=object).astype(np.complex128)
    data["nd"]=np.array(data["nd"],dtype=object).astype(np.complex128)
    return data

def AvgSigmajsonfileW(name):
    global omega,AvgSigmadat,Nfin,DEDargs,nd,pbar
    #print(str(nd),str(nd/Nfin))#,np.finfo(nd).precision,np.finfo(nd/Nfin).precision)
    #print('{0:.16f}'.format(nd[0]),'{0:.16f}'.format(nd[0]/Nfin[0]))
    data={"Ntot":pbar.total,"Nit": pbar.n,"poles": DEDargs[1],"U": DEDargs[2],"Sigma": DEDargs[3],"Ed": DEDargs[4],"Gamma": DEDargs[5],"ctype": DEDargs[6],"Edcalc": DEDargs[7],
    "Nimpurities": DEDargs[8],"U2": DEDargs[9],"J": DEDargs[10],"Tk": DEDargs[11],"etaco": DEDargs[12],
    "Nfin": Nfin,"omega": omega,"AvgSigmadat":[str(i) for i in (AvgSigmadat/Nfin[:,None]).squeeze()],"nd": [str(i) for i in (nd/Nfin)]}#np.real
    jsonObj=json.dumps(data, cls=NumpyArrayEncoder)
    with open(name, "w") as outfile:
        outfile.write(jsonObj)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

#    DEDargs=[N,poles,U,Sigma,Ed,Gamma,ctype,Edcalc,Nimpurities,U2,J,Tk,etaco,SizeO,bound,posb,log,base]
def valinit():
    global omega,Npoles,c,pbar,eta,Hn,n,AvgSigmadat,Nfin,nd,DEDargs,Lor
    if DEDargs[16]: omega,Npoles=np.concatenate((-np.logspace(np.log(DEDargs[14])/np.log(DEDargs[17]),np.log(1e-5)/np.log(DEDargs[17]),int(np.round(DEDargs[13]/2)),base=DEDargs[17]),np.logspace(np.log(1e-5)/np.log(DEDargs[17]),np.log(DEDargs[14])/np.log(DEDargs[17]),int(np.round(DEDargs[13]/2)),base=DEDargs[17]))),int(DEDargs[1]/DEDargs[8])
    else: omega,Npoles=np.linspace(-DEDargs[14],DEDargs[14],DEDargs[13]),int(DEDargs[1]/DEDargs[8])
    c,pbar,eta=[Jordan_wigner_transform(i,2*DEDargs[1]) for i in range(2*DEDargs[1])],trange(DEDargs[0],position=DEDargs[15],leave=False,desc='Iterations',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),DEDargs[12][0]*abs(omega)+DEDargs[12][1]
    (Hn,n),AvgSigmadat,Nfin,nd=Operators(c,DEDargs[8],DEDargs[1]),np.zeros((len(DEDargs[11]),DEDargs[13]),dtype='complex_'),np.zeros(len(DEDargs[11]),dtype='float'),np.zeros(len(DEDargs[11]),dtype='complex_')
    Lor=Lorentzian(omega,DEDargs[5],DEDargs[1],DEDargs[4],DEDargs[3])[0]

def parainit():
    global DEDargs,app,number,pbar,Npoles,c,Hn,n,Lor
    if not started:
        try:
            if number<int(app.N_Entry.get()):
                DEDargs[0]=int(app.N_Entry.get())
                app.N_Entry.configure(state="disabled")
                #currentit=pbar.n
                #pbar.reset(DEDargs[0])
                #pbar.update(currentit)
                pbar.total=DEDargs[0]
                pbar.refresh()
            DEDargs[2]=float(app.U_Entry.get())
            DEDargs[3]=float(app.Sigma_Entry.get())
            DEDargs[4]=float(app.Ed_Entry.get())
            DEDargs[5]=float(app.Gamma_Entry.get())
            DEDargs[1]=int(app.scaling_optionemenu.get())
            app.progressbar_1.set(number/DEDargs[0])
            app.U_Entry.configure(state="disabled")
            app.Sigma_Entry.configure(state="disabled")
            app.Ed_Entry.configure(state="disabled")
            app.Gamma_Entry.configure(state="disabled")
            app.scaling_optionemenu.configure(state="disabled")
            app.sidebar_button_3.configure(state="disabled")
            Npoles,c=int(DEDargs[1]/DEDargs[8]),[Jordan_wigner_transform(i,2*DEDargs[1]) for i in range(2*DEDargs[1])]
            (Hn,n)=Operators(c,DEDargs[8],DEDargs[1])
            Lor=Lorentzian(omega,DEDargs[5],DEDargs[1],DEDargs[4],DEDargs[3])[0]
        except:
            pass

def showgraph():
    global omega,Lor,AvgSigmadat,Nfin,DEDargs,app
    matplotlib.rc('axes',edgecolor='white')

    fDOS=(-np.imag(np.nan_to_num(1/(omega-AvgSigmadat/Nfin[:,None]-DEDargs[4]+1j*DEDargs[5])))/np.pi).squeeze()
    fig=plt.figure(figsize=(7,5.6))#,facecolor='#242424')

    plt.rc('legend',fontsize=12)
    plt.rc('font',size=17)
    plt.rc('xtick',labelsize=17,color='white')
    plt.rc('ytick',labelsize=17,color='white')
    axis_font={'fontname':'Calibri','size':'17'}
    plt.xlim(min(omega),max(omega))
    plt.gca().set_ylim(bottom=0,top=1.2)
    plt.gca().set_xticks(np.linspace(min(omega),max(omega),2*int(max(omega))+1),minor=False)
    plt.xlabel("$\\omega$ [-]",**axis_font,color='white')
    plt.gca().set_ylabel("$\\rho$($\\omega$)",va="bottom",rotation=0,labelpad=30,**axis_font,color='white')
    plt.plot(omega,Lor,'--r',linewidth=4,label='$\\rho_0$')
    plt.plot(omega,fDOS,'-b',label='$\\rho,$n='+str(DEDargs[1]))
    plt.legend(fancybox=False).get_frame().set_edgecolor('black')
    plt.grid()
    plt.tight_layout()
    fig.set_facecolor("none")
    plt.gca().set_facecolor("#242424")

    app.plot_frame.configure(fg_color="transparent")
    canvas = FigureCanvasTkAgg(fig,master=app.plot_frame)
    s = tk.ttk.Style()
    bg = s.lookup("TFrame", "background")
    bg_16bit =app.winfo_rgb('DarkGray')#bg
    bg_string="#242424"
    canvas.get_tk_widget().config(bg=bg_string)
    canvas.draw()
    canvas.get_tk_widget().grid(row=1,column=1)#,sticky="nsew"    .get_tk_widget() row=8,columnspan=3
    app.plot_frame.grid_rowconfigure((0,2), weight=1)
    app.plot_frame.grid_columnconfigure((0,2), weight=1)
    #canvas.get_tk_widget().grid_rowconfigure(0, weight=0)
    #canvas.get_tk_widget().grid_columnconfigure(0, weight=0)

    #canvas.get_tk_widget().pack(side=tkinter.TOP,pady=10, padx=10,fill="y")
    #canvas.get_tk_widget().place(anchor=tkinter.e) #x=40,y=0, fill=tkinter.BOTH)
    app.plot_frame.update()
    plt.close()

def savegraph():
    global omega,Lor,AvgSigmadat,Nfin,DEDargs,app,Lor
    matplotlib.rc('axes',edgecolor='black')
    savedata()
    fDOS=(-np.imag(np.nan_to_num(1/(omega-AvgSigmadat/Nfin[:,None]-DEDargs[4]+1j*DEDargs[5])))/np.pi).squeeze()
    if app.entry_2.get().endswith(".json"):
        DOSplot(fDOS,Lor,omega,app.entry_2.get().replace(".json",""),'$\\rho,$n='+str(DEDargs[1]))
    else:
        app.entry_2.delete(0,last_index=tk.END)
        app.entry_2.insert(0, 'Try again')    



def savedata():
    global app, started,stopped
    if started or stopped:
        try:
            if not app.entry_2.get().endswith(".json"):
                app.entry_2.delete(0,last_index=tk.END)
                app.entry_2.insert(0, 'Try again')
            else:   
                AvgSigmajsonfileW(app.entry_2.get())
        except IOError:
            #savfilename()
            app.entry_2.delete(0,last_index=tk.END)
            app.entry_2.insert(0, 'Try again')

number=0
DEDargs=[200000]

class ProgressBar(customtkinter.CTkProgressBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # create the text item in the internal canvas
        self._canvas.create_text(0, 0, text=f"{number}/{DEDargs[0]}", fill="white",
                                 font=14, anchor="c", tags="progress_text")#self._variable.get()

    # override function to move the progress text at the center of the internal canvas
    def _update_dimensions_event(self, event):
        super()._update_dimensions_event(event)
        self._canvas.coords("progress_text", event.width/2, event.height/2)

    # override function to update the progress text whenever new value is set
    def set(self, val, **kwargs):
        super().set(val, **kwargs)
        self._canvas.itemconfigure("progress_text", text=f"{number}/{DEDargs[0]}")

def CenterWindowToDisplay(Screen:customtkinter.CTk,width:int,height:int,scale_factor:float=1.0):
    #return f"{width}x{height}+{int(0.75*(Screen.winfo_screenwidth()-width)-10)}+{int(0.75*(Screen.winfo_screenheight()-height)-46)}"
    return f"{width}x{height}+{int((0.5*(Screen.winfo_screenwidth()-width)-7)*scale_factor)}+{int((0.5*(Screen.winfo_screenheight()-height)-31)*scale_factor)}"

def DEDUI():
    global paused,started,stopped,istar,number,app,DEDargs
    paused=False
    started=False
    stopped=False
    istar=0
    number=0

    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("blue")
    app=customtkinter.CTk()
    app.title("Distributional Exact Diagonalization AIM simulator")
    app_width,app_height=1100,580
    app.geometry(CenterWindowToDisplay(app,app_width,app_height,app._get_window_scaling()))
    #print(app.winfo_screenwidth(),app.winfo_screenheight(),screeninfo.get_monitors()[0].width,screeninfo.get_monitors()[0].height)

    #x,y=(app.winfo_screenwidth()/2)-(app_width/2),(app.winfo_screenheight()/2)-(app_height/2)
    #x,y=(screeninfo.get_monitors()[0].width/2)-(app_width/2),(screeninfo.get_monitors()[0].height/2)-(app_height/2)
    #app.geometry(f"{app_width}x{app_height}+{int(x)}+{int(y)}")

    #app.geometry(f"{1100}x{580}")
    #app.eval('tk::PlaceWindow . center')
    #app.after(201, lambda :app.iconbitmap('DEDicon.ico'))
    app.iconbitmap('DEDicon.ico')
    app.resizable(width=False, height=False)

    app.grid_columnconfigure(1, weight=1)
    app.grid_columnconfigure((2,3), weight=0)
    app.grid_rowconfigure((0, 1,2), weight=1)

    app.sidebar_frame = customtkinter.CTkFrame(app, width=140, corner_radius=0)
    app.sidebar_frame.grid(row=0, column=0, rowspan=5, sticky="nsew")
    app.sidebar_frame.grid_rowconfigure(14, weight=1)
    app.logo_label = customtkinter.CTkLabel(app.sidebar_frame, text="AIM Parameters", font=customtkinter.CTkFont(size=20, weight="bold"))
    app.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
    app.N_label = customtkinter.CTkLabel(app.sidebar_frame,text="No. of iterations DED:", anchor="w")
    app.N_label.grid(row=1, column=0, padx=20, pady=(5, 0))
    app.N_Entry=customtkinter.CTkEntry(app.sidebar_frame,placeholder_text="200000")
    app.N_Entry.grid(row=2, column=0, padx=20, pady=(5, 0))
    app.U_label=customtkinter.CTkLabel(app.sidebar_frame,text="Coulomb repulsion U:", anchor="w")
    app.U_label.grid(row=3, column=0, padx=20, pady=(5, 0))
    app.U_Entry=customtkinter.CTkEntry(app.sidebar_frame,placeholder_text="3")
    app.U_Entry.grid(row=4, column=0, padx=20, pady=(5, 0))
    app.Sigma_label=customtkinter.CTkLabel(app.sidebar_frame,text="Effective one-body potential \u03A3\N{SUBSCRIPT ZERO}:", anchor="w")
    app.Sigma_label.grid(row=5, column=0, padx=20, pady=(5, 0))
    app.Sigma_Entry=customtkinter.CTkEntry(app.sidebar_frame,placeholder_text="1.5")
    app.Sigma_Entry.grid(row=6, column=0, padx=20, pady=(5, 0))
    app.Ed_label=customtkinter.CTkLabel(app.sidebar_frame,text="Electron impurity energy \u03B5d:", anchor="w")
    app.Ed_label.grid(row=7, column=0, padx=20, pady=(5, 0))
    app.Ed_Entry=customtkinter.CTkEntry(app.sidebar_frame,placeholder_text="-1.5")
    app.Ed_Entry.grid(row=8, column=0, padx=20, pady=(5, 0))
    app.Gamma_label=customtkinter.CTkLabel(app.sidebar_frame,text="Flat hybridization constant \u0393:", anchor="w")
    app.Gamma_label.grid(row=9, column=0, padx=20, pady=(5, 0))
    app.Gamma_Entry=customtkinter.CTkEntry(app.sidebar_frame,placeholder_text="0.3")
    app.Gamma_Entry.grid(row=10, column=0, padx=20, pady=(5, 0))
    app.scaling_label = customtkinter.CTkLabel(app.sidebar_frame, text="No. of poles:", anchor="w")
    app.scaling_label.grid(row=11, column=0, padx=20, pady=(5, 0))

    app.scaling_optionemenu = customtkinter.CTkOptionMenu(app.sidebar_frame, values=["2", "3", "4", "5", "6"])
    app.scaling_optionemenu.grid(row=12, column=0, padx=20, pady=(5, 0))

    app.sidebar_button_3 = customtkinter.CTkButton(app.sidebar_frame,text="Submit Parameters",command=parainit)
    app.sidebar_button_3.grid(row=13, column=0, padx=20, pady=(40, 40))
    #app.sidebar_button_3.configure(state="disabled")

    app.N_Entry.insert(0,str(DEDargs[0]))
    app.U_Entry.insert(0,str(DEDargs[2]))
    app.Sigma_Entry.insert(0,str(DEDargs[3]))
    app.Ed_Entry.insert(0,str(DEDargs[4]))
    app.Gamma_Entry.insert(0,str(DEDargs[5]))
    app.scaling_optionemenu.set(str(DEDargs[1]))

    app.entry = customtkinter.CTkEntry(app, placeholder_text="C:\\")
    app.entry.grid(row=3, column=1, columnspan=2, padx=(20, 20), pady=(20, 0), sticky="nsew")
    app.main_button = customtkinter.CTkButton(master=app,text="Submit file to load",fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"),command=fileloader)
    app.main_button.grid(row=3, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")

    app.entry_2 = customtkinter.CTkEntry(app, placeholder_text="example.json")
    app.entry_2.grid(row=4, column=1, columnspan=2, padx=(20, 20), pady=10, sticky="nsew")
    app.main_button_2 = customtkinter.CTkButton(app,text="Submit save location",fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"),command=savfilename)
    app.main_button_2.grid(row=4, column=3, padx=(20, 20), pady=10, sticky="nsew")




    app.plot_frame=customtkinter.CTkFrame(app,width=250,height=374)
    app.plot_frame.grid(row=0, column=1, columnspan=2,rowspan=1,padx=(20, 0), pady=(20, 10), sticky="nsew")



    app.slider_progressbar_frame = customtkinter.CTkFrame(app)#, fg_color="transparent"
    app.slider_progressbar_frame.grid(row=1, column=1, columnspan=2,padx=(20, 0), pady=(5, 0), sticky="nsew")
    #app.slider_progressbar_frame.grid_columnconfigure((0,1,2), weight=0)
    #app.slider_progressbar_frame.grid_rowconfigure(1, weight=0)

    app.slider_progressbar_frame.grid_columnconfigure((0,1,2,3,4), weight=1)

    #app.progressbar_1 = customtkinter.CTkProgressBar(app.slider_progressbar_frame)#,width=400)
    #prog_num = ctk.IntVar(value=50)
    #count_var=tkinter.StringVar(value=f"{0}/{DEDargs[0]}")
    count_var = customtkinter.IntVar(value=0)

    app.progressbar_1 = ProgressBar(master=app.slider_progressbar_frame,variable=count_var,height=30)#app.slider_progressbar_frame

    app.progressbar_1.grid(row=0, column=0, columnspan=5,padx=20, pady=(5, 0), sticky="nsew")
    app.progressbar_1.set(0)


    #count_var=tkinter.StringVar(value="test")
    #app.count_text=customtkinter.CTkLabel(app.slider_progressbar_frame,text=f"{0}/{DEDargs[0]}")
    #app.count_text.configure()
    #app.count_text.grid(row=0, column=4,padx=20,pady=(5, 0))

    app.start_button=customtkinter.CTkButton(app.slider_progressbar_frame,text="Start",command=startDED)
    app.start_button.grid(row=1, column=0,padx=10,pady=(5, 0))
    app.pause_button=customtkinter.CTkButton(app.slider_progressbar_frame,text="Pause",command=pauseDED)
    app.pause_button.grid(row=1, column=1,padx=10,pady=(5, 0))
    app.stop_button=customtkinter.CTkButton(app.slider_progressbar_frame,text="Stop",command=stopDED)
    app.stop_button.grid(row=1, column=2,padx=10,pady=(5, 0))
    app.show_button=customtkinter.CTkButton(app.slider_progressbar_frame,text="Show Results",command=showgraph)#,state="disabled"
    app.show_button.grid(row=1, column=3,padx=10,pady=(5, 0))
    app.save_button=customtkinter.CTkButton(app.slider_progressbar_frame,text="Save Graph",command=savegraph)#,state="disabled"
    app.save_button.grid(row=1, column=4,padx=10,pady=(5, 0))

    app.settings_tab=customtkinter.CTkTabview(app, width=80)
    app.settings_tab.grid(row=0, column=3, rowspan=2,padx=(20, 20), pady=(20, 0), sticky="nsew")
    app.settings_tab.add("CTkTabview")
    app.settings_tab.add("Tab 2")
    app.settings_tab.add("Tab 3")
    app.settings_tab.tab("CTkTabview").grid_columnconfigure(0, weight=1)
    app.settings_tab.tab("Tab 2").grid_columnconfigure(0, weight=1)

    app.protocol('WM_DELETE_WINDOW', ask_question)
    return app
    #self.scrollable_frame = customtkinter.CTkScrollableFrame(self, label_text="CTkScrollableFrame")

def ask_question():
    global app
    msg = CTkMessagebox(master=app,title="Exit?", message="Do you want to close the program?",
                        icon="question", option_1="Cancel", option_2="No", option_3="Yes")
    response = msg.get()
    if response=="Yes":
        app.destroy()       


def main(N=200000,poles=2,U=3,Sigma=3/2,Ed=-3/2,Gamma=0.3,SizeO=1001,etaco=[0.02,1e-39],ctype='n',Edcalc='',bound=3,Tk=[0],Nimpurities=1,U2=0,J=0,posb=1,log=False,base=1.5):
    global DEDargs
    #omega,selectpcT,selectpT,Npoles,c,pbar,eta,Hn,n,AvgSigmadat,Nfin,nd,Lor
    DEDargs=[N,poles,U,Sigma,Ed,Gamma,ctype,Edcalc,Nimpurities,U2,J,Tk,etaco,SizeO,bound,posb,log,base]
    valinit()
    DEDUI().mainloop()

main()