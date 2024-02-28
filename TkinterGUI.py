import tkinter as tk
import customtkinter as ctk
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
from CTkMessagebox import CTkMessagebox
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from DEDlib import *
import warnings
warnings.filterwarnings("ignore",category=RuntimeWarning)
from tempfile import TemporaryFile
import json
from json import JSONEncoder

def valinit():
    global omega,Npoles,c,pbar,eta,Hn,n,AvgSigmadat,Nfin,nd,DEDargs,Lor
    if DEDargs[16]: omega,Npoles=np.concatenate((-np.logspace(np.log(DEDargs[14])/np.log(DEDargs[17]),np.log(1e-5)/np.log(DEDargs[17]),int(np.round(DEDargs[13]/2)),base=DEDargs[17]),np.logspace(np.log(1e-5)/np.log(DEDargs[17]),np.log(DEDargs[14])/np.log(DEDargs[17]),int(np.round(DEDargs[13]/2)),base=DEDargs[17]))),int(DEDargs[1]/DEDargs[8])
    else: omega,Npoles=np.linspace(-DEDargs[14],DEDargs[14],DEDargs[13]),int(DEDargs[1]/DEDargs[8])
    c,pbar,eta=[Jordan_wigner_transform(i,2*DEDargs[1]) for i in range(2*DEDargs[1])],trange(DEDargs[0],position=DEDargs[15],leave=False,desc='Iterations',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),DEDargs[12][0]*abs(omega)+DEDargs[12][1]
    (Hn,n),AvgSigmadat,Nfin,nd=Operators(c,DEDargs[8],DEDargs[1]),np.zeros((len(DEDargs[11]),DEDargs[13]),dtype='complex_'),np.zeros(len(DEDargs[11]),dtype='float'),np.zeros(len(DEDargs[11]),dtype='complex_')
    Lor=Lorentzian(omega,DEDargs[5],DEDargs[1],DEDargs[4],DEDargs[3])[0]

def iterationDED(reset=False):
    global omega,Npoles,c,pbar,eta,Hn,n,AvgSigmadat,Nfin,nd
    while not reset:
        NewM,nonG,_=Startrans(Npoles,np.sort(Lorentzian(omega,DEDargs[5],Npoles,DEDargs[4],DEDargs[3])[1]),omega,eta)
        H0,H=HamiltonianAIM(np.repeat(NewM[0][0],DEDargs[8]),np.tile([NewM[k+1][k+1] for k in range(len(NewM)-1)],(DEDargs[8],1)),np.tile(NewM[0,1:],(DEDargs[8],1)),DEDargs[2],DEDargs[3],DEDargs[9],DEDargs[10],Hn)
        try: (MBGdat,Boltzmann,Ev0),reset=Constraint(DEDargs[6],H0,H,omega,eta,c,n,DEDargs[11],np.array([ar<DEDargs[0] for ar in Nfin]))
        except (np.linalg.LinAlgError,ValueError,scipy.sparse.linalg.ArpackNoConvergence): (MBGdat,Boltzmann,Ev0),reset=(np.zeros(len(omega),dtype='complex_'),np.zeros(len(DEDargs[11])),np.array([])),False
        if np.isnan(1/nonG-1/MBGdat+DEDargs[3]).any() or np.array([i>=1000 for i in np.real(1/nonG-1/MBGdat+DEDargs[3])]).any(): reset=False
    Nfin,AvgSigmadat,nd=Nfin+Boltzmann,AvgSigmadat+(1/nonG-1/MBGdat+DEDargs[3])*Boltzmann[:,None],nd+np.conj(Ev0).T@sum(Hn[0]).data.tocoo()@Ev0*Boltzmann
    if DEDargs[6]=='sn': pbar.n+=1
    else: pbar.n=int(min(Nfin))
    pbar.refresh()

def parainit():
    global DEDargs,app,number,pbar,Npoles,c,Hn,n,Lor
    if not started:
        try:
            if number<int(app.N_Entry.get()):
                pbar.total=DEDargs[0]=int(app.N_Entry.get())
                app.N_Entry.configure(state="disabled")
                pbar.refresh()
            DEDargs[1:6]=[int(app.scaling_optionemenu.get()),float(app.U_Entry.get()),float(app.Sigma_Entry.get()),float(app.Ed_Entry.get()),float(app.Gamma_Entry.get())]
            app.progressbar_1.set(number/DEDargs[0])
            app.U_Entry.configure(state="disabled")
            app.Sigma_Entry.configure(state="disabled")
            app.Ed_Entry.configure(state="disabled")
            app.Gamma_Entry.configure(state="disabled")
            app.scaling_optionemenu.configure(state="disabled")
            app.sidebar_button_3.configure(state="disabled")
            Npoles,c,Lor=int(DEDargs[1]/DEDargs[8]),[Jordan_wigner_transform(i,2*DEDargs[1]) for i in range(2*DEDargs[1])],Lorentzian(omega,DEDargs[5],DEDargs[1],DEDargs[4],DEDargs[3])[0]
            (Hn,n)=Operators(c,DEDargs[8],DEDargs[1])
        except: pass

def counter():
    global paused,stopped,started,DEDargs,pbar,Nfin,number
    if not paused and pbar.n!=DEDargs[0]:
        iterationDED()
        number=pbar.n
        app.progressbar_1.set(pbar.n/DEDargs[0])
    if not stopped and pbar.n!=DEDargs[0]: app.after(1,counter)
    elif pbar.n==DEDargs[0]: pbar.close()
    else:
        started,number,pbar.n=False,0,0
        app.progressbar_1.set(0)
        pbar.reset()

def startDED():
    global started,paused,number,istar,pbar,stopped,app
    istar+=1
    if istar==1:
        if not started: 
            started,stopped,pbar.start_t=True,False,pbar._time()
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
    if started: paused=not paused 

class NumpyArrayEncoder(JSONEncoder):
    def default(self,obj):
        if isinstance(obj,np.ndarray): return obj.tolist()
        return JSONEncoder.default(self,obj)

def AvgSigmajsonfileW(name):
    global omega,AvgSigmadat,Nfin,DEDargs,nd,pbar
    data={"Ntot":pbar.total,"Nit":pbar.n,"poles":DEDargs[1],"U":DEDargs[2],"Sigma":DEDargs[3],"Ed":DEDargs[4],"Gamma":DEDargs[5],"ctype":DEDargs[6],"Edcalc":DEDargs[7],"Nimpurities":DEDargs[8],"U2":DEDargs[9],"J":DEDargs[10],"Tk":DEDargs[11],"etaco":DEDargs[12],
    "Nfin": Nfin,"omega": omega,"AvgSigmadat":[str(i) for i in (AvgSigmadat/Nfin[:,None]).squeeze()],"nd": [str(i) for i in (nd/Nfin)]}
    jsonObj=json.dumps(data,cls=NumpyArrayEncoder)
    with open(name, "w") as outfile: outfile.write(jsonObj)

def savedata():
    global app,started,stopped
    if started or stopped:
        try:
            if not app.entry_2.get().endswith(".json"):
                app.entry_2.delete(0,last_index=tk.END)
                app.entry_2.insert(0,'Try again')
            else:   
                AvgSigmajsonfileW(app.entry_2.get())
        except IOError:
            app.entry_2.delete(0,last_index=tk.END)
            app.entry_2.insert(0,'Try again')

def stopDED():
    global stopped,started,paused,number,istar,omega,app
    savedata()
    if not stopped and started: stopped,paused,number,istar=True,False,0,0

def showgraph():
    global omega,Lor,AvgSigmadat,Nfin,DEDargs,app
    mpl.rc('axes',edgecolor='white')
    fig,axis_font,fDOS=plt.figure(figsize=(7,5.6)),{'fontname':'Calibri','size':'17'},(-np.imag(np.nan_to_num(1/(omega-AvgSigmadat/Nfin[:,None]-DEDargs[4]+1j*DEDargs[5])))/np.pi).squeeze()
    plt.rc('legend',fontsize=12)
    plt.rc('font',size=17)
    plt.rc('xtick',labelsize=17,color='white')
    plt.rc('ytick',labelsize=17,color='white')
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
    canvas=FigureCanvasTkAgg(fig,master=app.plot_frame)
    canvas.get_tk_widget().config(bg="#242424")
    canvas.draw()
    canvas.get_tk_widget().grid(row=1,column=1)
    app.plot_frame.grid_rowconfigure(0, weight=1)
    app.plot_frame.grid_rowconfigure(2, weight=1)
    app.plot_frame.grid_columnconfigure(0, weight=1)
    app.plot_frame.grid_columnconfigure(2, weight=1)
    app.plot_frame.update()
    plt.close()

def fileloader():
    global app,started,omega,AvgSigmadat,Nfin,Lor,DEDargs,eta,pbar,Npoles,Hn,n,nd,c,number
    if not started:
        try:
            data=AvgSigmajsonfileR(app.entry.get())
            Nfin,omega,AvgSigmadat,nd=np.array(data["Nfin"]),np.array(data["omega"]),np.array(data["AvgSigmadat"]*data["Nfin"]).squeeze(),np.array(np.array(data["nd"],dtype=np.complex128)*np.array(data["Nfin"],dtype=np.float64),dtype=np.complex128)
            DEDargs=[data["Ntot"],data["poles"],data["U"],data["Sigma"],data["Ed"],data["Gamma"],data["ctype"],data["Edcalc"],data["Nimpurities"],data["U2"],data["J"],data["Tk"],data["etaco"]]
            pbar.total,pbar.n=DEDargs[0],data["Nit"]
            number=pbar.n
            app.progressbar_1.set(pbar.n/DEDargs[0])
            Lor,Npoles,c,eta=Lorentzian(omega,DEDargs[5],DEDargs[1],DEDargs[4],DEDargs[3])[0],int(DEDargs[1]/DEDargs[8]),[Jordan_wigner_transform(i,2*DEDargs[1]) for i in range(2*DEDargs[1])],DEDargs[12][0]*abs(omega)+DEDargs[12][1]
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
        except IOError or FileNotFoundError:
            app.entry.delete(0,last_index=tk.END)
            app.entry.insert(0,'Try again')

def AvgSigmajsonfileR(name):#dir
    data=json.load(open(name))
    data["AvgSigmadat"],data["nd"]=np.array(data["AvgSigmadat"],dtype=object).astype(np.complex128),np.array(data["nd"],dtype=object).astype(np.complex128)
    return data

def savfilename():
    global app
    if not app.entry_2.get().endswith(".json"):
        app.entry_2.delete(0,last_index=tk.END)
        app.entry_2.insert(0,'Try again')

class ProgressBar(ctk.CTkProgressBar):
    def __init__(self,itnum,Total, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self._canvas.create_text(0,0,text=f"{itnum}/{Total}",fill="white",font=14,anchor="c",tags="progress_text")

    def _update_dimensions_event(self, event):
        super()._update_dimensions_event(event)
        self._canvas.coords("progress_text",event.width/2,event.height/2)

    def set(self, val,itnum,Total, **kwargs):
        super().set(val,**kwargs)
        self._canvas.itemconfigure("progress_text",text=f"{number}/{DEDargs[0]}")

class mainApp(ctk.CTk):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.app_width,self.app_height=300,200
        self.x,self.y=(screeninfo.get_monitors()[0].width/2)-(self.app_width*0.77),(screeninfo.get_monitors()[0].height/2)-(self.app_height*0.82)
        self.geometry(f"{self.app_width}x{self.app_height}+{int(self.x)}+{int(self.y)}")
        #self.eval('tk::PlaceWindow . center')
        self.iconbitmap('DEDicon.ico')
        self.title("DED simulator menu")
        self.grid_rowconfigure((0,1,2),weight=0)
        self.grid_columnconfigure(0,weight=1)
        self.menu_label=ctk.CTkLabel(self,text="Choose simulation type",font=ctk.CTkFont(size=20,weight="bold"))
        self.menu_label.grid(row=0,column=0,padx=20,pady=(20,10))
        self.scaling_optionemenu=ctk.CTkOptionMenu(self,width=200,values=["SAIM single sim","Sampled poles Distr.","ASAIM single sim","GNR SAIM single sim","Impurity Entropy","Stdev calculator"])
        self.scaling_optionemenu.grid(row=1,column=0,padx=20,pady=(5,0))
        self.button_open=ctk.CTkButton(self,text="Open",command=lambda:self.open_toplevel(simsel=self.scaling_optionemenu.get()))
        self.button_open.grid(row=2,column=0,padx=20,pady=(20,20))
        self.top_level_windows={"SAIM single sim":SAIMWINDOW,"Sampled poles Distr.":polesWINDOW,"ASAIM single sim":ASAIMWINDOW,"GNR SAIM single sim":GNRWINDOW,"Impurity Entropy":EntropyWINDOW,"Stdev calculator":StdevWINDOW}

    def open_toplevel(self,simsel):
        self.toplevel_window=self.top_level_windows[simsel](selfroot=self)
        self.scaling_optionemenu.configure(state="disabled")
        self.button_open.configure(state="disabled")

class SAIMWINDOW(ctk.CTkToplevel):
    def __init__(self,selfroot,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.root=selfroot
        self.title("Distributional Exact Diagonalization AIM simulator")
        self.app_width,self.app_height=1100,580
        self.x,self.y=(screeninfo.get_monitors()[0].width/2)-(self.app_width*0.77),(screeninfo.get_monitors()[0].height/2)-(self.app_height*0.82)
        self.geometry(f"{self.app_width}x{self.app_height}+{int(self.x)}+{int(self.y)}")
        self.after(200, lambda: self.iconbitmap('DEDicon.ico'))
        self.resizable(width=False, height=False)
        self.after(100, self.focus)
        self.protocol('WM_DELETE_WINDOW', self.enableroot)

    def enableroot(self):
        self.root.scaling_optionemenu.configure(state="normal")
        self.root.button_open.configure(state="normal")
        self.destroy()

class polesWINDOW(ctk.CTkToplevel):
    pass

class ASAIMWINDOW(ctk.CTkToplevel):
    pass

class GNRWINDOW(ctk.CTkToplevel):
    pass

class EntropyWINDOW(ctk.CTkToplevel):
    pass

class StdevWINDOW(ctk.CTkToplevel):
    pass

mainApp().mainloop()