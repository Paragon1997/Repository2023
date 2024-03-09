## module TkinterGUI
''' TkinterGUI is the Graphical User Interface made with Tkinter for using the DEDlib tooling library'''

import tkinter as tk
import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from DEDlib import *
import warnings
import json
from json import JSONEncoder
from ast import literal_eval
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
warnings.filterwarnings("ignore",category=RuntimeWarning)

#status bar in window
#prevent cross sim loading
#add ask window main root window
#other main functions in seperate windows (graphene, S etc.)
#make exe with pyinstaller

class NumpyArrayEncoder(JSONEncoder):
    """``NumpyArrayEncoder(JSONEncoder)``.\n
Encodes Numpy array for ``json.dumps()``."""
    def default(self,obj):
        if isinstance(obj,np.ndarray): return obj.tolist()
        return JSONEncoder.default(self,obj)
    
def AvgSigmajsonfileR(name):
    """``AvgSigmajsonfileR(name)``.\n
Loads ``.json`` file to collect DED data from previous simulation session."""
    data=json.load(open(name))
    data["AvgSigmadat"],data["nd"]=np.array(data["AvgSigmadat"],dtype=object).astype(np.complex128),np.array(data["nd"],dtype=object).astype(np.complex128)
    return data

def AvgSigmajsonfileW(root,name):
    """``AvgSigmajsonfileW(root,name)``.\n
Writes ``.json`` file including DED simulation settings and collected data."""
    if root.DEDargs[7]=='AS':root.fDOS=(-np.imag(np.nan_to_num(1/(root.omega-root.AvgSigmadat/root.Nfin[:,None]+(root.AvgSigmadat[:,int(np.round(root.DEDargs[13]/2))]/root.Nfin)[:,None]+1j*root.DEDargs[5])))/np.pi).squeeze()
    else:root.fDOS=(-np.imag(np.nan_to_num(1/(root.omega-root.AvgSigmadat/root.Nfin[:,None]-root.DEDargs[4]+1j*root.DEDargs[5])))/np.pi).squeeze()
    data={"Ntot":root.pbar.total,"Nit":root.pbar.n,"telapsed":root.pbar.format_dict["elapsed"],"poles":root.DEDargs[1],"U":root.DEDargs[2],"Sigma":root.DEDargs[3],"Ed":root.DEDargs[4],"Gamma":root.DEDargs[5],"ctype":root.DEDargs[6],"Edcalc":root.DEDargs[7],"Nimpurities":root.DEDargs[8],"U2":root.DEDargs[9],"J":root.DEDargs[10],"Tk":root.DEDargs[11],"etaco":root.DEDargs[12],"SizeO":root.DEDargs[13],"bound":root.DEDargs[14],"posb":root.DEDargs[15],"log":root.DEDargs[16],"base":root.DEDargs[17],
    "Nfin":root.Nfin,"omega":root.omega,"fDOS":root.fDOS,"AvgSigmadat":[str(i) for i in (root.AvgSigmadat/root.Nfin[:,None]).squeeze()],"nd":[str(i) for i in (root.nd/root.Nfin)]}
    jsonObj=json.dumps(data,cls=NumpyArrayEncoder)
    with open(name,"w") as outfile: outfile.write(jsonObj)

def savfilename(root,entry):
    """``savfilename(root,entry)``.\n
Checks whether filename input in entry is valid for use in constructing ``.json`` file."""
    if not entry.get().endswith(".json"):
        entry.delete(0,last_index=tk.END)
        entry.insert(0,'Try again')
    _=savedata(root,entry)

def savedata(root,entry):
    """``savedata(root,entry)``.\n
Saves DED results based on current simulation data."""
    if root.started or root.stopped:
        try:
            if not entry.get().endswith(".json"):
                entry.delete(0,last_index=tk.END)
                entry.insert(0,'Try again')
                return False
            else:   
                AvgSigmajsonfileW(root,entry.get())
                return True
        except IOError:
            entry.delete(0,last_index=tk.END)
            entry.insert(0,'Try again')
            return False
    else: return False

def savegraph(root,entry):
    """``savegraph(root,entry)``.\n
Draws and saves DOS graph including the interacting DOS data acquired from the current DED simulation results."""
    mpl.rc('axes',edgecolor='black')
    _=savedata(root,entry)
    if root.DEDargs[7]=='AS':root.fDOS=(-np.imag(np.nan_to_num(1/(root.omega-root.AvgSigmadat/root.Nfin[:,None]+(root.AvgSigmadat[:,int(np.round(root.DEDargs[13]/2))]/root.Nfin)[:,None]+1j*root.DEDargs[5])))/np.pi).squeeze()
    else:root.fDOS=(-np.imag(np.nan_to_num(1/(root.omega-root.AvgSigmadat/root.Nfin[:,None]-root.DEDargs[4]+1j*root.DEDargs[5])))/np.pi).squeeze()
    if entry.get().endswith(".json"):
        try:
            DOSplot(root.fDOS,root.Lor,root.omega,entry.get().replace(".json",""),root.graphlegend_Entry.get(),log=bool(root.graphlogy_checkbox.get()),ymax=float(root.graphymax_Entry.get()))
            if root.DEDargs[16]:
                DOSxlogplot(root.fDOS,root.Lor,root.omega,entry.get().replace(".json","")+"logx",root.graphlegend_Entry.get(),ymax=float(root.graphymax_Entry.get()),incneg=True)
                DOSxlogplot(root.fDOS,root.Lor,root.omega,entry.get().replace(".json","")+"logxpos",root.graphlegend_Entry.get(),ymax=float(root.graphymax_Entry.get()),incneg=False)
        except: pass
    else:
        entry.delete(0,last_index=tk.END)
        entry.insert(0,'Try again') 

class ProgressBar(ctk.CTkProgressBar):
    """``ProgressBar(ctk.CTkProgressBar)``.\n
Custom progressbar with current (and total) number of iterations displayed on the bar."""
    def __init__(self,itnum,Total,*args,**kwargs):
        self.itnum,self.Total=itnum,Total
        super().__init__(*args,**kwargs)
        self._canvas.create_text(0,0,text=f"{itnum}/{Total}",fill="white",font=14,anchor="c",tags="progress_text")

    def _update_dimensions_event(self,event):
        super()._update_dimensions_event(event)
        self._canvas.coords("progress_text",event.width/2,event.height/2)

    def set(self,val,**kwargs):
        super().set(val,**kwargs)
        self._canvas.itemconfigure("progress_text",text=f"{self.itnum}/{self.Total}")

def CenterWindowToDisplay(Screen:ctk.CTk,width:int,height:int,scale_factor:float=1.0):
    """``CenterWindowToDisplay(Screen:ctk.CTk,width:int,height:int,scale_factor:float=1.0)``.\n
Calculates the necessary coordinates for a ``customTkinter`` window to be displayed in the center of the screen."""
    return f"{width}x{height}+{int((0.5*(Screen.winfo_screenwidth()-width)-7)*scale_factor)}+{int((0.5*(Screen.winfo_screenheight()-height)-31)*scale_factor)}"

class mainApp(ctk.CTk):
    """``mainApp(ctk.CTk)``.\n
The main ``customTkinter`` class for the DED Anderson impurity model simulator application window."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.app_width,self.app_height=300,200
        self.geometry(CenterWindowToDisplay(self,self.app_width,self.app_height,self._get_window_scaling()))
        self.iconbitmap('DEDicon.ico')
        self.resizable(width=False,height=False)
        self.title("DED simulator menu")
        self.focus()
        self.grid_rowconfigure((0,1,2),weight=0)
        self.grid_columnconfigure(0,weight=1)
        self.menu_label=ctk.CTkLabel(self,text="Choose simulation type",font=ctk.CTkFont(size=20,weight="bold"))
        self.menu_label.grid(row=0,column=0,padx=20,pady=(20,10))
        self.scaling_optionemenu=ctk.CTkOptionMenu(self,width=200,values=["SAIM single sim","Sampled poles Distr.","ASAIM single sim","GNR SAIM single sim","Impurity Entropy","Stdev calculator"])
        self.scaling_optionemenu.grid(row=1,column=0,padx=20,pady=(5,0))
        self.button_open=ctk.CTkButton(self,text="Open",command=lambda:self.open_toplevel(simsel=self.scaling_optionemenu.get()))
        self.button_open.grid(row=2,column=0,padx=20,pady=(20,20))
        self.top_level_windows={"SAIM single sim":SAIMWINDOW,"Sampled poles Distr.":polesWINDOW,"ASAIM single sim":ASAIMWINDOW,"GNR SAIM single sim":GNRWINDOW,"Impurity Entropy":EntropyWINDOW,"Stdev calculator":StdevWINDOW}
        self.protocol('WM_DELETE_WINDOW',self.quitApp)

    def open_toplevel(self,simsel):
        """``open_toplevel(self,simsel)``.\n
    Class method to initialize selected window for specific simulation type."""
        self.toplevel_window=self.top_level_windows[simsel](selfroot=self)
        self.after(100,self.lower)
        self.scaling_optionemenu.configure(state="disabled")
        self.button_open.configure(state="disabled")

    def quitApp(self):
        """``quitApp(self)``.\n
    Class method to show messagebox when attempting to close main window."""
        self.msg=CTkMessagebox(master=self,title="Exit?",message="Are you sure you want to quit the entire application?",icon="question",option_1="Cancel",option_2="No",option_3="Yes")
        if self.msg.get()=="Yes": self.destroy()

class SAIMWINDOW(ctk.CTkToplevel):
    """``SAIMWINDOW(ctk.CTkToplevel)``.\n
Class for Symmetric Anderson impurity model DED simmulation window."""
    def __init__(self,selfroot,N=200000,poles=4,U=3,Sigma=3/2,Ed=-3/2,Gamma=0.3,SizeO=1001,etaco=[0.02,1e-39],ctype='n',Edcalc='',bound=3,Tk=[0],Nimpurities=1,U2=0,J=0,posb=1,log=False,base=1.5,ymax=1.2,logy=False,fDOScolor='b',*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.root,self.paused,self.started,self.stopped,self.loaded,self.parainitialized,self.poleDOS,self.telapsed,self.DEDargs=selfroot,False,False,False,False,False,False,0,[N,poles,U,Sigma,Ed,Gamma,ctype,Edcalc,Nimpurities,U2,J,Tk,etaco,SizeO,bound,posb,log,base,ymax,logy,fDOScolor]
        if self.DEDargs[16]:self.omega,self.Npoles=np.concatenate((-np.logspace(np.log(self.DEDargs[14])/np.log(self.DEDargs[17]),np.log(1e-5)/np.log(self.DEDargs[17]),int(np.round(self.DEDargs[13]/2)),base=self.DEDargs[17]),np.logspace(np.log(1e-5)/np.log(self.DEDargs[17]),np.log(self.DEDargs[14])/np.log(self.DEDargs[17]),int(np.round(self.DEDargs[13]/2)),base=self.DEDargs[17]))),int(self.DEDargs[1]/self.DEDargs[8])
        else:self.omega,self.Npoles=np.linspace(-self.DEDargs[14],self.DEDargs[14],self.DEDargs[13]),int(self.DEDargs[1]/self.DEDargs[8])
        self.c,self.pbar,self.eta=[Jordan_wigner_transform(i,2*self.DEDargs[1]) for i in range(2*self.DEDargs[1])],trange(self.DEDargs[0],position=self.DEDargs[15],leave=False,desc='Iterations',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),self.DEDargs[12][0]*abs(self.omega)+self.DEDargs[12][1]
        (self.Hn,self.n),self.AvgSigmadat,self.Nfin,self.nd,self.Lor=Operators(self.c,self.DEDargs[8],self.DEDargs[1]),np.zeros((len(self.DEDargs[11]),self.DEDargs[13]),dtype='complex_'),np.zeros(len(self.DEDargs[11]),dtype='float'),np.zeros(len(self.DEDargs[11]),dtype='complex_'),Lorentzian(self.omega,self.DEDargs[5],self.DEDargs[1],self.DEDargs[4],self.DEDargs[3])[0]
        self.title("Distributional Exact Diagonalization AIM simulator")
        self.app_width,self.app_height=1100,580
        self.geometry(CenterWindowToDisplay(self,self.app_width,self.app_height,self._get_window_scaling()))
        self.after(200,lambda:self.iconbitmap('DEDicon.ico'))
        self.resizable(width=False,height=False)
        self.after(200,self.focus)
        self.grid_columnconfigure(1,weight=1)
        self.grid_columnconfigure((2,3,4),weight=0)
        self.grid_rowconfigure((0,1,2),weight=1)
        self.sidebar_frame=ctk.CTkFrame(self,width=140,corner_radius=0)
        self.sidebar_frame.grid(row=0,column=0,rowspan=5,sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(16,weight=1)
        self.logo_label=ctk.CTkLabel(self.sidebar_frame,text="AIM Parameters",font=ctk.CTkFont(size=20,weight="bold"))
        self.logo_label.grid(row=0,column=0,padx=20,pady=(20,10))
        self.N_label=ctk.CTkLabel(self.sidebar_frame,text="No. of iterations DED:",anchor="w")
        self.N_label.grid(row=1,column=0,padx=20,pady=(5,0))
        self.N_Entry=ctk.CTkEntry(self.sidebar_frame,placeholder_text="200000")
        self.N_Entry.grid(row=2,column=0,padx=20,pady=(0,0))
        self.U_label=ctk.CTkLabel(self.sidebar_frame,text="Coulomb repulsion U:",anchor="w")
        self.U_label.grid(row=3,column=0,padx=20,pady=(5,0))
        self.U_Entry=ctk.CTkEntry(self.sidebar_frame,placeholder_text="3")
        self.U_Entry.grid(row=4,column=0,padx=20,pady=(0,0))
        self.Sigma_label=ctk.CTkLabel(self.sidebar_frame,text="Effective one-body potential \u03A3\N{SUBSCRIPT ZERO}:",anchor="w")
        self.Sigma_label.grid(row=5,column=0,padx=20,pady=(5,0))
        self.Sigma_Entry=ctk.CTkEntry(self.sidebar_frame,placeholder_text="1.5")
        self.Sigma_Entry.grid(row=6,column=0,padx=20,pady=(0,0))
        self.Ed_label=ctk.CTkLabel(self.sidebar_frame,text="Electron impurity energy \u03B5d:",anchor="w")
        self.Ed_label.grid(row=7,column=0,padx=20,pady=(5,0))
        self.Ed_Entry=ctk.CTkEntry(self.sidebar_frame,placeholder_text="-1.5")
        self.Ed_Entry.grid(row=8,column=0,padx=20,pady=(0,0))
        self.Gamma_label=ctk.CTkLabel(self.sidebar_frame,text="Flat hybridization constant \u0393:",anchor="w")
        self.Gamma_label.grid(row=9,column=0,padx=20,pady=(5,0))
        self.Gamma_Entry=ctk.CTkEntry(self.sidebar_frame,placeholder_text="0.3")
        self.Gamma_Entry.grid(row=10,column=0,padx=20,pady=(0,0))
        self.scaling_label=ctk.CTkLabel(self.sidebar_frame,text="No. of poles:",anchor="w")
        self.scaling_label.grid(row=11,column=0,padx=20,pady=(5,0))
        self.scaling_optionemenu=ctk.CTkOptionMenu(self.sidebar_frame,values=["2","3","4","5","6"])
        self.scaling_optionemenu.grid(row=12,column=0,padx=20,pady=(0,0))
        self.Tk_label=ctk.CTkLabel(self.sidebar_frame,text="Simulated Temperatures kbT:",anchor="w")
        self.Tk_label.grid(row=13,column=0,padx=20,pady=(5,0))
        self.Tk_Entry=ctk.CTkEntry(self.sidebar_frame,placeholder_text="[0]")
        self.Tk_Entry.grid(row=14,column=0,padx=20,pady=(0,0))
        self.sidebar_button_3=ctk.CTkButton(self.sidebar_frame,text="Submit Parameters",command=self.parainit)
        self.sidebar_button_3.grid(row=15,column=0,padx=20,pady=(15,15))
        self.N_Entry.insert(0,str(self.DEDargs[0]))
        self.U_Entry.insert(0,str(self.DEDargs[2]))
        self.Sigma_Entry.insert(0,str(self.DEDargs[3]))
        self.Ed_Entry.insert(0,str(self.DEDargs[4]))
        self.Gamma_Entry.insert(0,str(self.DEDargs[5]))
        self.scaling_optionemenu.set(str(self.DEDargs[1]))
        self.Tk_Entry.insert(0,str(self.DEDargs[11]))
        self.entry=ctk.CTkEntry(self,placeholder_text="C:\\")
        self.entry.grid(row=3,column=1,columnspan=2,padx=(20,0),pady=(10,5),sticky="nsew")
        self.openfile_button=ctk.CTkButton(self,text="Open file",fg_color="transparent",width=100,border_width=2,text_color=("gray10","#DCE4EE"),command=self.openfile)
        self.openfile_button.grid(row=3,column=3,padx=(20,5),pady=(10,5),sticky="w")
        self.main_button=ctk.CTkButton(self,text="Submit file to load",fg_color="transparent",width=150,border_width=2,text_color=("gray10","#DCE4EE"),command=self.fileloader)
        self.main_button.grid(row=3,column=4,padx=(0,20),pady=(10,5),sticky="e")
        self.entry_2=ctk.CTkEntry(self,placeholder_text="example.json")
        self.entry_2.grid(row=4,column=1,columnspan=2,padx=(20,0),pady=(5,10),sticky="nsew")
        self.saveasfile_button=ctk.CTkButton(self,text="Save as file",fg_color="transparent",width=100,border_width=2,text_color=("gray10","#DCE4EE"),command=self.saveasfile)
        self.saveasfile_button.grid(row=4,column=3,padx=(20,5),pady=(5,10),sticky="w")
        self.main_button_2=ctk.CTkButton(self,text="Submit file to save",fg_color="transparent",width=150,border_width=2,text_color=("gray10","#DCE4EE"),command=lambda:savfilename(self,self.entry_2))
        self.main_button_2.grid(row=4,column=4,padx=(0,20),pady=(5,10),sticky='e')
        self.plot_frame=ctk.CTkFrame(self,width=250,height=380)
        self.plot_frame.grid(row=0,column=1,columnspan=2,rowspan=1,padx=(20,0),pady=(20,10),sticky="nsew")
        self.slider_progressbar_frame=ctk.CTkFrame(self)
        self.slider_progressbar_frame.grid(row=1,column=1,columnspan=2,padx=(20,0),pady=(5,0),sticky="nsew")
        self.slider_progressbar_frame.grid_columnconfigure((0,1,2,3,4,5),weight=1)
        self.progressbar_1=ProgressBar(master=self.slider_progressbar_frame,variable=ctk.IntVar(value=0),itnum=self.pbar.n,Total=self.pbar.total,height=30)
        self.progressbar_1.grid(row=0,column=0,columnspan=6,padx=20,pady=(5,0),sticky="nsew")
        self.start_button=ctk.CTkButton(self.slider_progressbar_frame,text="Start",command=self.startDED)
        self.start_button.grid(row=1,column=0,padx=4,pady=(5,10))
        self.pause_button=ctk.CTkButton(self.slider_progressbar_frame,text="Pause",command=self.pauseDED)
        self.pause_button.grid(row=1,column=1,padx=4,pady=(5,10))
        self.stop_button=ctk.CTkButton(self.slider_progressbar_frame,text="Stop",command=self.stopDED)
        self.stop_button.grid(row=1,column=2,padx=4,pady=(5,10))
        self.show_button=ctk.CTkButton(self.slider_progressbar_frame,text="Show Graph",command=self.showgraph)
        self.show_button.grid(row=1,column=3,padx=4,pady=(5,10))
        self.save_button=ctk.CTkButton(self.slider_progressbar_frame,text="Save Graph",command=lambda:savegraph(self,self.entry_2))
        self.save_button.grid(row=1,column=4,padx=4,pady=(5,10))
        self.reset_button=ctk.CTkButton(self.slider_progressbar_frame,text="Reset",command=self.resetDED)
        self.reset_button.grid(row=1,column=5,padx=4,pady=(5,10))
        self.pause_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.reset_button.configure(state="disabled")
        self.settings_tab=ctk.CTkTabview(self, width=261)
        self.settings_tab.grid(row=0,column=3,rowspan=2,columnspan=2,padx=(20,20),pady=(20,0),sticky="nsew")
        self.settings_tab.add("Adv.")
        self.settings_tab.add("Multi orb.")
        self.settings_tab.add("Graph")
        self.settings_tab.tab("Adv.").grid_columnconfigure(0,weight=1)
        self.settings_tab.tab("Multi orb.").grid_columnconfigure(0,weight=1)
        self.settings_tab.tab("Graph").grid_columnconfigure(0,weight=1)
        self.scrollable_tab=ctk.CTkScrollableFrame(self.settings_tab.tab("Adv."),height=350,width=220,label_text="Advanced settings",label_font=ctk.CTkFont(size=20,weight="bold"),label_fg_color="transparent")
        self.scrollable_tab.grid(row=0, column=0,rowspan=2, padx=(0, 0), pady=(0, 0))
        self.eta_label=ctk.CTkLabel(self.scrollable_tab,text="Imaginary part of frequency arg. of\nGreen's function \u03B7 ([slope,offset]):",anchor="w")
        self.eta_label.grid(row=0,column=0,padx=10,pady=(5,0))
        self.eta_Entry=ctk.CTkEntry(self.scrollable_tab,placeholder_text="[0.02, 1e-39]")
        self.eta_Entry.grid(row=1,column=0,padx=10,pady=(2,0))
        self.eta_Entry.insert(0,str(self.DEDargs[12]))
        self.SizeO_label=ctk.CTkLabel(self.scrollable_tab,text="No. of energies in spectrum \u03C9:",anchor="w")
        self.SizeO_label.grid(row=2,column=0,padx=10,pady=(5,0))
        self.SizeO_Entry=ctk.CTkEntry(self.scrollable_tab,placeholder_text="1001")
        self.SizeO_Entry.grid(row=3,column=0,padx=10,pady=(0,0))
        self.SizeO_Entry.insert(0,str(self.DEDargs[13]))
        self.bound_label=ctk.CTkLabel(self.scrollable_tab,text="Range energies in spectrum \u03C9:",anchor="w")
        self.bound_label.grid(row=4,column=0,padx=10,pady=(5,0))
        self.bound_Entry=ctk.CTkEntry(self.scrollable_tab,placeholder_text="3")
        self.bound_Entry.grid(row=5,column=0,padx=10,pady=(0,0))
        self.bound_Entry.insert(0,str(self.DEDargs[14]))
        self.ctype_label=ctk.CTkLabel(self.scrollable_tab,text="Constraint type:",anchor="w")
        self.ctype_label.grid(row=6,column=0,padx=10,pady=(5,0))
        self.ctype_optionemenu=ctk.CTkOptionMenu(self.scrollable_tab,values=["n"," ","n%2","sn","ssn","dn","mosn"])
        self.ctype_optionemenu.grid(row=7,column=0,padx=10,pady=(0,0))
        self.log_checkbox=ctk.CTkCheckBox(master=self.scrollable_tab,text="Logarithmic scale energies \u03C9")
        self.log_checkbox.grid(row=8,column=0,padx=10,pady=(10,0))
        self.base_label=ctk.CTkLabel(self.scrollable_tab,text="Base log. scale:",anchor="w")
        self.base_label.grid(row=9,column=0,padx=10,pady=(5,0))
        self.base_Entry=ctk.CTkEntry(self.scrollable_tab,placeholder_text="1.5")
        self.base_Entry.grid(row=10,column=0,padx=10,pady=(0,0))
        self.base_Entry.insert(0,str(self.DEDargs[17]))
        self.Edcalc_label=ctk.CTkLabel(self.scrollable_tab,text="\u03B5d in interacting DOS:",anchor="w")
        self.Edcalc_label.grid(row=11,column=0,padx=10,pady=(5,0))
        self.Edcalc_optionemenu=ctk.CTkOptionMenu(self.scrollable_tab,values=["","AS"])
        self.Edcalc_optionemenu.grid(row=12,column=0,padx=10,pady=(0,0))
        self.tab_label2=ctk.CTkLabel(self.settings_tab.tab("Multi orb."),text="Multi orb. settings",font=ctk.CTkFont(size=20,weight="bold"))
        self.tab_label2.grid(row=0,column=0,padx=10,pady=(10,10))
        self.Nimpurities_label=ctk.CTkLabel(self.settings_tab.tab("Multi orb."),text="No. of orbitals:",anchor="w")
        self.Nimpurities_label.grid(row=1,column=0,padx=10,pady=(5,0))
        self.Nimpurities_optionemenu=ctk.CTkOptionMenu(self.settings_tab.tab("Multi orb."),values=["1","2"])
        self.Nimpurities_optionemenu.grid(row=2,column=0,padx=10,pady=(0,0))
        self.U2_label=ctk.CTkLabel(self.settings_tab.tab("Multi orb."),text="Inter-orbital Coulomb repulsion U':",anchor="w")
        self.U2_label.grid(row=3,column=0,padx=10,pady=(5,0))
        self.U2_Entry=ctk.CTkEntry(self.settings_tab.tab("Multi orb."),placeholder_text="0")
        self.U2_Entry.grid(row=4,column=0,padx=10,pady=(0,0))
        self.U2_Entry.insert(0,str(self.DEDargs[9]))
        self.J_label=ctk.CTkLabel(self.settings_tab.tab("Multi orb."),text="Hund’s rule coupling JH:",anchor="w")
        self.J_label.grid(row=5,column=0,padx=10,pady=(5,0))
        self.J_Entry=ctk.CTkEntry(self.settings_tab.tab("Multi orb."),placeholder_text="0")
        self.J_Entry.grid(row=6,column=0,padx=10,pady=(0,0))
        self.J_Entry.insert(0,str(self.DEDargs[10]))
        self.tab_label3=ctk.CTkLabel(self.settings_tab.tab("Graph"),text="Graph settings",font=ctk.CTkFont(size=20,weight="bold"))
        self.tab_label3.grid(row=0,column=0,padx=10,pady=(10,10))
        self.graphlegend_label=ctk.CTkLabel(self.settings_tab.tab("Graph"),text="Legend label:",anchor="w")
        self.graphlegend_label.grid(row=1,column=0,padx=10,pady=(5,0))
        self.graphlegend_Entry=ctk.CTkEntry(self.settings_tab.tab("Graph"),width=200,placeholder_text='$\\rho_{constr.},N,$n='+str(self.DEDargs[1]))
        self.graphlegend_Entry.grid(row=2,column=0,padx=10,pady=(0,0))
        self.graphlegend_Entry.insert(0,'$\\rho_{constr.},N,$n='+str(self.DEDargs[1]))
        self.graphymax_label=ctk.CTkLabel(self.settings_tab.tab("Graph"),text="Limit y-axis:",anchor="w")
        self.graphymax_label.grid(row=3,column=0,padx=10,pady=(5,0))
        self.graphymax_Entry=ctk.CTkEntry(self.settings_tab.tab("Graph"),placeholder_text='1.2')
        self.graphymax_Entry.grid(row=4,column=0,padx=10,pady=(0,0))
        self.graphymax_Entry.insert(0,str(self.DEDargs[18]))
        self.graphlogy_checkbox=ctk.CTkCheckBox(master=self.settings_tab.tab("Graph"),text="Logarithmic y scale")
        self.graphlogy_checkbox.grid(row=5,column=0,padx=10,pady=(10,0))
        self.graphfDOScolor_label=ctk.CTkLabel(self.settings_tab.tab("Graph"),text="Interacting DOS color/style:",anchor="w")
        self.graphfDOScolor_label.grid(row=6,column=0,padx=10,pady=(5,0))
        self.graphfDOScolor_Entry=ctk.CTkEntry(self.settings_tab.tab("Graph"),placeholder_text='b')
        self.graphfDOScolor_Entry.grid(row=7,column=0,padx=10,pady=(0,0))
        self.graphfDOScolor_Entry.insert(0,str(self.DEDargs[20]))
        self.protocol('WM_DELETE_WINDOW',self.enableroot)

    def enableroot(self):
        """``enableroot(self)``.\n
    Class method to show messagebox when attempting to close ``SAIMWINDOW(ctk.CTkToplevel)`` window."""
        self.msg=CTkMessagebox(master=self,title="Exit?",message="Do you want to close the program?\nUnsaved progress will be lost.",icon="question",option_1="Cancel",option_2="No",option_3="Yes")
        if self.msg.get()=="Yes":
            self.root.scaling_optionemenu.configure(state="normal")
            self.root.button_open.configure(state="normal")
            self.lift(self.root)
            self.destroy()

    def parainit(self):
        """``parainit(self)``.\n
    Class method to save input parameters of DED from inputs in the ``SAIMWINDOW(ctk.CTkToplevel)`` window."""
        if not self.started:
            try:
                if self.pbar.n<int(self.N_Entry.get()):
                    self.pbar.total=self.DEDargs[0]=int(self.N_Entry.get())
                    self.N_Entry.configure(state="disabled")
                    self.start_button.configure(state="normal")
                    self.pbar.refresh()
                else:
                    self.N_Entry.delete(0,last_index=tk.END)
                    self.N_Entry.insert(0,str(self.pbar.n))
                self.parainitialized,self.DEDargs[1:],self.progressbar_1.Total,self.progressbar_1.itnum=True,[int(self.scaling_optionemenu.get()),float(self.U_Entry.get()),float(self.Sigma_Entry.get()),float(self.Ed_Entry.get()),float(self.Gamma_Entry.get()),self.ctype_optionemenu.get(),self.Edcalc_optionemenu.get(),int(self.Nimpurities_optionemenu.get()),float(self.U2_Entry.get()),float(self.J_Entry.get()),literal_eval(self.Tk_Entry.get()),literal_eval(self.eta_Entry.get()),int(self.SizeO_Entry.get()),float(self.bound_Entry.get()),self.DEDargs[15],bool(self.log_checkbox.get()),float(self.base_Entry.get()),float(self.graphymax_Entry.get()),bool(self.graphlogy_checkbox.get()),self.graphfDOScolor_Entry.get()],self.DEDargs[0],self.pbar.n
                self.progressbar_1.set(self.pbar.n/self.DEDargs[0])
                self.U_Entry.configure(state="disabled")
                self.Sigma_Entry.configure(state="disabled")
                self.Ed_Entry.configure(state="disabled")
                self.Gamma_Entry.configure(state="disabled")
                self.scaling_optionemenu.configure(state="disabled")
                self.Tk_Entry.configure(state="disabled")
                self.sidebar_button_3.configure(state="disabled")
                self.eta_Entry.configure(state="disabled")
                self.SizeO_Entry.configure(state="disabled")
                self.bound_Entry.configure(state="disabled")
                self.ctype_optionemenu.configure(state="disabled")
                self.log_checkbox.configure(state="disabled")
                self.base_Entry.configure(state="disabled")
                self.Edcalc_optionemenu.configure(state="disabled")
                self.Nimpurities_optionemenu.configure(state="disabled")
                self.U2_Entry.configure(state="disabled")
                self.J_Entry.configure(state="disabled")
                if not self.loaded:
                    if self.DEDargs[16]: self.omega,self.Npoles=np.concatenate((-np.logspace(np.log(self.DEDargs[14])/np.log(self.DEDargs[17]),np.log(1e-5)/np.log(self.DEDargs[17]),int(np.round(self.DEDargs[13]/2)),base=self.DEDargs[17]),np.logspace(np.log(1e-5)/np.log(self.DEDargs[17]),np.log(self.DEDargs[14])/np.log(self.DEDargs[17]),int(np.round(self.DEDargs[13]/2)),base=self.DEDargs[17]))),int(self.DEDargs[1]/self.DEDargs[8])
                    else: self.omega,self.Npoles=np.linspace(-self.DEDargs[14],self.DEDargs[14],self.DEDargs[13]),int(self.DEDargs[1]/self.DEDargs[8])
                    self.eta,self.AvgSigmadat,self.Nfin,self.nd,self.Npoles,self.c,self.Lor=self.DEDargs[12][0]*abs(self.omega)+self.DEDargs[12][1],np.zeros((len(self.DEDargs[11]),self.DEDargs[13]),dtype='complex_'),np.zeros(len(self.DEDargs[11]),dtype='float'),np.zeros(len(self.DEDargs[11]),dtype='complex_'),int(self.DEDargs[1]/self.DEDargs[8]),[Jordan_wigner_transform(i,2*self.DEDargs[1]) for i in range(2*self.DEDargs[1])],Lorentzian(self.omega,self.DEDargs[5],self.DEDargs[1],self.DEDargs[4],self.DEDargs[3])[0]
                    (self.Hn,self.n)=Operators(self.c,self.DEDargs[8],self.DEDargs[1])
            except: pass

    def saveasfile(self):
        """``saveasfile(self)``.\n
    Class method to choose save file location for a DED data file in File Explorer."""
        self.savefile=ctk.filedialog.asksaveasfilename(initialdir="",title="Save DED Data JSON File as",filetypes=[('JSON files','*.json')])
        self.root.lower()
        if self.savefile:
            self.entry_2.delete(0,last_index=tk.END)
            if not self.savefile.endswith(".json"): self.savefile+=".json"
            self.entry_2.insert(0,self.savefile)
            _=savedata(self,self.entry_2)

    def openfile(self):
        """``openfile(self)``.\n
    Class method to choose a DED data file to load in File Explorer."""
        self.loadfile=ctk.filedialog.askopenfilename(initialdir="",title="Select a DED Data JSON File",filetypes=[('JSON files','*.json')])
        self.root.lower()
        if self.loadfile:
            self.entry.delete(0,last_index=tk.END)
            self.entry.insert(0,self.loadfile)

    def fileloader(self):
        """``fileloader(self)``.\n
    Class method to load ``.json`` file and save data and settings from that particular simulation session to utilize for current session."""
        if not self.started:
            try:
                self.data=AvgSigmajsonfileR(self.entry.get())
                self.Nfin,self.omega,self.AvgSigmadat,self.nd=np.array(self.data["Nfin"]),np.array(self.data["omega"]),np.array(self.data["AvgSigmadat"]*self.data["Nfin"]).squeeze(),np.array(np.array(self.data["nd"],dtype=np.complex128)*np.array(self.data["Nfin"],dtype=np.float64),dtype=np.complex128)
                self.DEDargs=[self.data["Ntot"],self.data["poles"],self.data["U"],self.data["Sigma"],self.data["Ed"],self.data["Gamma"],self.data["ctype"],self.data["Edcalc"],self.data["Nimpurities"],self.data["U2"],self.data["J"],self.data["Tk"],self.data["etaco"],self.data["SizeO"],self.data["bound"],self.data["posb"],self.data["log"],self.data["base"],self.DEDargs[18],self.DEDargs[19],self.DEDargs[20]]
                self.progressbar_1.Total,self.progressbar_1.itnum=self.pbar.total,self.pbar.n=self.DEDargs[0],self.data["Nit"]
                self.progressbar_1.set(self.pbar.n/self.DEDargs[0])
                self.eta,self.Npoles,self.c,self.Lor=self.DEDargs[12][0]*abs(self.omega)+self.DEDargs[12][1],int(self.DEDargs[1]/self.DEDargs[8]),[Jordan_wigner_transform(i,2*self.DEDargs[1]) for i in range(2*self.DEDargs[1])],Lorentzian(self.omega,self.DEDargs[5],self.DEDargs[1],self.DEDargs[4],self.DEDargs[3])[0]
                (self.Hn,self.n),self.loaded,self.telapsed=Operators(self.c,self.DEDargs[8],self.DEDargs[1]),True,self.data["telapsed"]
                self.pbar.refresh()
                self.U_Entry.delete(0,last_index=tk.END)
                self.U_Entry.insert(0,str(self.DEDargs[2]))
                self.Sigma_Entry.delete(0,last_index=tk.END)
                self.Sigma_Entry.insert(0,str(self.DEDargs[3]))
                self.Ed_Entry.delete(0,last_index=tk.END)
                self.Ed_Entry.insert(0,str(self.DEDargs[4]))
                self.Gamma_Entry.delete(0,last_index=tk.END)
                self.Gamma_Entry.insert(0,str(self.DEDargs[5]))
                self.scaling_optionemenu.set(str(self.DEDargs[1]))
                self.Tk_Entry.delete(0,last_index=tk.END)
                self.Tk_Entry.insert(0,str(self.DEDargs[11]))
                self.eta_Entry.delete(0,last_index=tk.END)
                self.eta_Entry.insert(0,str(self.DEDargs[12]))
                self.SizeO_Entry.delete(0,last_index=tk.END)
                self.SizeO_Entry.insert(0,str(self.DEDargs[13]))
                self.bound_Entry.delete(0,last_index=tk.END)
                self.bound_Entry.insert(0,str(self.DEDargs[14]))
                self.ctype_optionemenu.set(str(self.DEDargs[6]))
                self.log_checkbox.configure(variable=tk.IntVar(value=int(self.DEDargs[16])))
                self.base_Entry.delete(0,last_index=tk.END)
                self.base_Entry.insert(0,str(self.DEDargs[17]))
                self.Edcalc_optionemenu.set(str(self.DEDargs[7]))
                self.Nimpurities_optionemenu.set(str(self.DEDargs[8]))
                self.U2_Entry.delete(0,last_index=tk.END)
                self.U2_Entry.insert(0,str(self.DEDargs[9]))
                self.J_Entry.delete(0,last_index=tk.END)
                self.J_Entry.insert(0,str(self.DEDargs[10]))
                if self.pbar.n>self.DEDargs[0]:
                    self.N_Entry.delete(0,last_index=tk.END)
                    self.N_Entry.insert(0,str(self.pbar.n))
                else:
                    self.N_Entry.delete(0,last_index=tk.END)
                    self.N_Entry.insert(0,str(self.DEDargs[0]))   
                if self.pbar.n==self.DEDargs[0]: self.start_button.configure(state="disabled")             
                self.U_Entry.configure(state="disabled")
                self.Sigma_Entry.configure(state="disabled")
                self.Ed_Entry.configure(state="disabled")
                self.Gamma_Entry.configure(state="disabled")
                self.scaling_optionemenu.configure(state="disabled")
                self.Tk_Entry.configure(state="disabled")
                self.eta_Entry.configure(state="disabled")
                self.SizeO_Entry.configure(state="disabled")
                self.bound_Entry.configure(state="disabled")
                self.ctype_optionemenu.configure(state="disabled")
                self.log_checkbox.configure(state="disabled")
                self.base_Entry.configure(state="disabled")
                self.Edcalc_optionemenu.configure(state="disabled")
                self.Nimpurities_optionemenu.configure(state="disabled")
                self.U2_Entry.configure(state="disabled")
                self.J_Entry.configure(state="disabled")
            except IOError or FileNotFoundError:
                self.entry.delete(0,last_index=tk.END)
                self.entry.insert(0,'Try again')
    
    def startDED(self):
        """``startDED(self)``.\n
    Class method to start DED calculations by initiating loop."""
        if not self.parainitialized: self.parainit()
        if self.parainitialized:
            self.start_button.configure(state="disabled")
            self.started,self.stopped,self.pbar.start_t=True,False,self.pbar._time()-self.telapsed
            self.loopDED()
            self.pause_button.configure(state="normal")
            self.stop_button.configure(state="normal")
            self.reset_button.configure(state="normal")
            self.main_button.configure(state="disabled")

    def loopDED(self):
        """``loopDED(self)``.\n
    Class method of the main SAIM DED loop which repeats iterations by repeatedly executing ``loopDED(self)``."""
        if not self.stopped and not self.paused and self.pbar.n!=self.DEDargs[0]:
            self.iterationDED()
            self.progressbar_1.itnum=self.pbar.n
            self.progressbar_1.set(self.pbar.n/self.DEDargs[0])
        if not self.stopped and self.pbar.n<self.DEDargs[0]:
            if self.DEDargs[1]>5:
                if self.pbar.n%100==0 and self.pbar.total>100:
                    _=savedata(self,self.entry_2)
                    self.showgraph()
                self.after(100,self.loopDED)
            else: self.after(1,self.loopDED)
        elif self.pbar.n>=self.DEDargs[0]: 
            self.pbar.close()
            self.stopDED()
            self.showgraph()
            savegraph(self,self.entry_2)
        elif not self.parainitialized: self.stopped=False

    def iterationDED(self,reset=False):
        """``iterationDED(self,reset=False)``.\n
    Class method of a single DED iteration which updates ``self.AvgSigmadat`` and other results for each iteration."""
        while not reset:
            self.NewM,self.nonG,_=Startrans(self.Npoles,np.sort(Lorentzian(self.omega,self.DEDargs[5],self.Npoles,self.DEDargs[4],self.DEDargs[3])[1]),self.omega,self.eta)
            self.H0,self.H=HamiltonianAIM(np.repeat(self.NewM[0][0],self.DEDargs[8]),np.tile([self.NewM[k+1][k+1] for k in range(len(self.NewM)-1)],(self.DEDargs[8],1)),np.tile(self.NewM[0,1:],(self.DEDargs[8],1)),self.DEDargs[2],self.DEDargs[3],self.DEDargs[9],self.DEDargs[10],self.Hn)
            try: (self.MBGdat,self.Boltzmann,self.Ev0),reset=Constraint(self.DEDargs[6],self.H0,self.H,self.omega,self.eta,self.c,self.n,self.DEDargs[11],np.array([ar<self.DEDargs[0] for ar in self.Nfin]),self.poleDOS)
            except (np.linalg.LinAlgError,ValueError,scipy.sparse.linalg.ArpackNoConvergence): (self.MBGdat,self.Boltzmann,self.Ev0),reset=(np.zeros(len(self.omega),dtype='complex_'),np.zeros(len(self.DEDargs[11])),np.array([])),False
            if np.isnan(1/self.nonG-1/self.MBGdat+self.DEDargs[3]).any() or np.array([i>=1000 for i in np.real(1/self.nonG-1/self.MBGdat+self.DEDargs[3])]).any(): reset=False
        self.Nfin,self.AvgSigmadat,self.nd=self.Nfin+self.Boltzmann,self.AvgSigmadat+(1/self.nonG-1/self.MBGdat+self.DEDargs[3])*self.Boltzmann[:,None],self.nd+np.conj(self.Ev0).T@sum(self.Hn[0]).data.tocoo()@self.Ev0*self.Boltzmann
        if self.DEDargs[6]=='sn':self.pbar.n+=1
        else: self.pbar.n=int(min(self.Nfin))
        self.pbar.refresh()

    def pauseDED(self):
        """``pauseDED(self)``.\n
    Class method which pauses DED calculations while executing ``loopDED(self)``."""
        if self.started: self.paused=not self.paused

    def stopDED(self):
        """``stopDED(self)``.\n
    Class method which stops DED calculations in ``loopDED(self)`` and saves progress."""
        if savedata(self,self.entry_2) and not self.stopped and self.started:
            self.stopped,self.paused,self.started=True,False,False
            self.pause_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")
            if self.pbar.n!=self.DEDargs[0]: self.start_button.configure(state="normal")
        elif not self.stopped and not self.paused: self.pauseDED()

    def showgraph(self):
        """``showgraph(self)``.\n
    Class method to show graph of current results based on finished iterations."""
        try:
            mpl.rc('axes',edgecolor='white')
            if self.DEDargs[7]=='AS':self.fig,self.axis_font,self.fDOS=plt.figure(figsize=(9.5,7.6),dpi=50*self._get_window_scaling()),{'fontname':'Calibri','size':'19'},(-np.imag(np.nan_to_num(1/(self.omega-self.AvgSigmadat/self.Nfin[:,None]+(self.AvgSigmadat[:,int(np.round(self.DEDargs[13]/2))]/self.Nfin)[:,None]+1j*self.DEDargs[5])))/np.pi).squeeze()
            else:self.fig,self.axis_font,self.fDOS=plt.figure(figsize=(9.5,7.6),dpi=50*self._get_window_scaling()),{'fontname':'Calibri','size':'19'},(-np.imag(np.nan_to_num(1/(self.omega-self.AvgSigmadat/self.Nfin[:,None]-self.DEDargs[4]+1j*self.DEDargs[5])))/np.pi).squeeze()
            plt.rc('legend',fontsize=14)
            plt.rc('font',size=19)
            plt.rc('xtick',labelsize=17,color='white')
            plt.rc('ytick',labelsize=17,color='white')
            plt.xlim(min(self.omega),max(self.omega))
            if not bool(self.graphlogy_checkbox.get()):
                plt.gca().set_ylim(bottom=0,top=float(self.graphymax_Entry.get()))
                plt.gca().set_xticks(np.linspace(min(self.omega),max(self.omega),2*int(max(self.omega))+1),minor=False)
            else:
                plt.yscale('log')
                plt.gca().set_ylim(bottom=0.0001,top=float(self.graphymax_Entry.get()))
                plt.gca().set_xticks(np.linspace(min(self.omega),max(self.omega),int(max(self.omega))+int(max(self.omega))%2+1),minor=False)     
            plt.xlabel("$\\omega$ [-]",**self.axis_font,color='white')
            plt.gca().set_ylabel("$\\rho$($\\omega$)",va="bottom",rotation=0,labelpad=30,**self.axis_font,color='white')
            plt.plot(self.omega,self.Lor,'--r',linewidth=4,label='$\\rho_0$')
            plt.plot(self.omega,self.fDOS,self.graphfDOScolor_Entry.get(),label=self.graphlegend_Entry.get())
            plt.legend(fancybox=False).get_frame().set_edgecolor('black')
            plt.grid()
            plt.tight_layout()
            self.fig.set_facecolor("none")
            plt.gca().set_facecolor("#242424")
            self.plot_frame.configure(fg_color="transparent")
            self.canvas=FigureCanvasTkAgg(self.fig,master=self.plot_frame)
            self.canvas.get_tk_widget().config(bg="#242424")
            self.canvas.draw()
            self.canvas.get_tk_widget().grid(row=1,column=1)
            self.plot_frame.grid_rowconfigure((0,2),weight=1)
            self.plot_frame.grid_columnconfigure((0,2),weight=1)
            self.plot_frame.update()
        except: pass
        plt.close()

    def resetDED(self):
        """``resetDED(self)``.\n
    Class method to reset DED calculation progress in ``loopDED(self)``."""
        self.pbar.reset()
        if self.loaded:
            self.progressbar_1.itnum=self.pbar.n=self.data["Nit"]
            self.Nfin,self.AvgSigmadat,self.nd,self.telapsed=np.array(self.data["Nfin"]),np.array(self.data["AvgSigmadat"]*self.data["Nfin"]).squeeze(),np.array(np.array(self.data["nd"],dtype=np.complex128)*np.array(self.data["Nfin"],dtype=np.float64),dtype=np.complex128),self.data["telapsed"]
        else: 
            self.progressbar_1.itnum=self.pbar.n=0
            self.AvgSigmadat,self.Nfin,self.nd,self.telapsed=np.zeros((len(self.DEDargs[11]),self.DEDargs[13]),dtype='complex_'),np.zeros(len(self.DEDargs[11]),dtype='float'),np.zeros(len(self.DEDargs[11]),dtype='complex_'),0
        self.progressbar_1.set(self.pbar.n/self.DEDargs[0])
        self.pbar.refresh()
        self.paused,self.started,self.stopped,self.loaded,self.parainitialized=False,False,True,False,False
        self.start_button.configure(state="normal")
        self.pause_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.reset_button.configure(state="disabled")
        self.main_button.configure(state="normal")
        self.N_Entry.configure(state="normal")
        self.U_Entry.configure(state="normal")
        self.Sigma_Entry.configure(state="normal")
        self.Ed_Entry.configure(state="normal")
        self.Gamma_Entry.configure(state="normal")
        self.scaling_optionemenu.configure(state="normal")
        self.Tk_Entry.configure(state="normal")
        self.eta_Entry.configure(state="normal")
        self.SizeO_Entry.configure(state="normal")
        self.bound_Entry.configure(state="normal")
        self.ctype_optionemenu.configure(state="normal")
        self.log_checkbox.configure(state="normal")
        self.base_Entry.configure(state="normal")
        self.Edcalc_optionemenu.configure(state="normal")
        self.Nimpurities_optionemenu.configure(state="normal")
        self.U2_Entry.configure(state="normal")
        self.J_Entry.configure(state="normal")

class polesWINDOW(ctk.CTkToplevel):
    def __init__(self,selfroot,N=200000,poles=4,U=3,Sigma=3/2,Ed=-3/2,Gamma=0.3,SizeO=1001,etaco=[0.02,1e-39],ctype='n',Edcalc='',bound=3,Tk=[0],Nimpurities=1,U2=0,J=0,posb=1,log=False,base=1.5,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.root,self.paused,self.started,self.stopped,self.loaded,self.poleDOS,self.telapsed,self.DEDargs=selfroot,False,False,False,False,True,0,[N,poles,U,Sigma,Ed,Gamma,ctype,Edcalc,Nimpurities,U2,J,Tk,etaco,SizeO,bound,posb,log,base]
        if self.DEDargs[16]:self.omega,self.Npoles=np.concatenate((-np.logspace(np.log(self.DEDargs[14])/np.log(self.DEDargs[17]),np.log(1e-5)/np.log(self.DEDargs[17]),int(np.round(self.DEDargs[13]/2)),base=self.DEDargs[17]),np.logspace(np.log(1e-5)/np.log(self.DEDargs[17]),np.log(self.DEDargs[14])/np.log(self.DEDargs[17]),int(np.round(self.DEDargs[13]/2)),base=self.DEDargs[17]))),int(self.DEDargs[1]/self.DEDargs[8])
        else:self.omega,self.Npoles=np.linspace(-self.DEDargs[14],self.DEDargs[14],self.DEDargs[13]),int(self.DEDargs[1]/self.DEDargs[8])
        self.selectpcT,self.c,self.pbar,self.eta=[],[Jordan_wigner_transform(i,2*self.DEDargs[1]) for i in range(2*self.DEDargs[1])],trange(self.DEDargs[0],position=self.DEDargs[15],leave=False,desc='Iterations',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),self.DEDargs[12][0]*abs(self.omega)+self.DEDargs[12][1]
        (self.Hn,self.n),self.Lor=Operators(self.c,self.DEDargs[8],self.DEDargs[1]),Lorentzian(self.omega,self.DEDargs[5],self.DEDargs[1],self.DEDargs[4],self.DEDargs[3])[0]
        self.title("Distributional Exact Diagonalization AIM simulator")
        self.app_width,self.app_height=1100,580
        self.geometry(CenterWindowToDisplay(self,self.app_width,self.app_height,self._get_window_scaling()))
        self.after(200,lambda:self.iconbitmap('DEDicon.ico'))
        self.resizable(width=False,height=False)
        self.after(200,self.focus)
        self.grid_columnconfigure(1,weight=1)
        self.grid_columnconfigure((2,3),weight=0)
        self.grid_rowconfigure((0,1,2),weight=1)

    def itpolesDED(self,reset=False):
        while not reset:
            self.NewM,self.nonG,self.select=Startrans(self.Npoles,np.sort(Lorentzian(self.omega,self.DEDargs[5],self.Npoles,self.DEDargs[4],self.DEDargs[3])[1]),self.omega,self.eta)
            self.H0,self.H=HamiltonianAIM(np.repeat(self.NewM[0][0],self.DEDargs[8]),np.tile([self.NewM[k+1][k+1] for k in range(len(self.NewM)-1)],(self.DEDargs[8],1)),np.tile(self.NewM[0,1:],(self.DEDargs[8],1)),self.DEDargs[2],self.DEDargs[3],self.DEDargs[9],self.DEDargs[10],self.Hn)
            try: _,reset=Constraint(self.DEDargs[6],self.H0,self.H,self.omega,self.eta,self.c,self.n,self.DEDargs[11],np.array([ar<self.DEDargs[0] for ar in self.Nfin]),self.poleDOS)
            except (np.linalg.LinAlgError,ValueError,scipy.sparse.linalg.ArpackNoConvergence): pass
        self.selectpcT.append(self.select)
        if self.DEDargs[6]=='sn':self.pbar.n+=1
        else: self.pbar.n=int(min(self.Nfin))
        self.pbar.refresh()

class ASAIMWINDOW(ctk.CTkToplevel):
    pass

class GNRWINDOW(ctk.CTkToplevel):
    pass

class EntropyWINDOW(ctk.CTkToplevel):
    pass

class StdevWINDOW(ctk.CTkToplevel):
    pass

if __name__ == "__main__":
    mainApp().mainloop()