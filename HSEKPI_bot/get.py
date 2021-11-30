import streamlit as st
import psycopg2
#conn = sqlite3.connect('data.db', check_same_thread=False)
conn=psycopg2.connect("dbname='d47ep4809u8n1i' user='gnqjgldwuozvln' password='777ec0683293ba3a6e27fa9ce5e4a8622e38f5fec4c6207d4da51373b04d398e' host='ec2-23-20-124-77.compute-1.amazonaws.com' port='5432' ")
c = conn.cursor()
#=================================	

	
def get_Chantier(Chantier):
	c.execute('SELECT * FROM Accueil WHERE Chantier=%s',(Chantier))
	data = c.fetchall()
	return data	
	

def get_NArrivant(NArrivant):
	c.execute('SELECT * FROM Accueil WHERE NArrivant=%s',(NArrivant))
	data = c.fetchall()
	return data	
	
	
def get_Ninduction(Ninduction):
	c.execute('SELECT * FROM Accueil WHERE Ninduction=%s',(Ninduction))
	data = c.fetchall()
	return data	


#==========================================================

def get_Chantier(Chantier):
	c.execute('SELECT * FROM TBM WHERE Chantier=%s',(Chantier))
	data = c.fetchall()
	return data
	
def get_NChantier(NChantier):
	c.execute('SELECT * FROM TBM WHERE NChantier=%s',(NChantier))
	data = c.fetchall()
	return data	
	

def get_NArrivant(NTBM):
	c.execute('SELECT * FROM TBM WHERE NTBM=%s',(NTBM))
	data = c.fetchall()
	return data	
	

#===========================

def get_Chantier(Chantier):
	c.execute('SELECT * FROM NC WHERE Chantier=%s',(Chantier))
	data = c.fetchall()
	return data
	

def get_NCR(NCR):
	c.execute('SELECT * FROM NC WHERE NCR=%s',(NCR))
	data = c.fetchall()
	return data


def get_FNCR(FNCR):
	c.execute('SELECT * FROM NC WHERE FNCR=%s',(FNCR))
	data = c.fetchall()
	return data
	
	
def get_NCC(NCC):
	c.execute('SELECT * FROM NC WHERE NCC=%s',(NCC))
	data = c.fetchall()
	return data

	
def get_FNCC(FNCC):
	c.execute('SELECT * FROM NC WHERE FNCC=%s',(FNCC))
	data = c.fetchall()
	return data


#========================================================================

def get_Chantier(Chantier):
	c.execute('SELECT * FROM Changements WHERE Chantier=%s',(Chantier))
	data = c.fetchall()
	return data
	
	
def get_NCH(NCH):
	c.execute('SELECT * FROM Changements WHERE NCH=%s',(NCH))
	data = c.fetchall()
	return data

	
def get_FNCH(FNCH):
	c.execute('SELECT * FROM Changements WHERE FNCH=%s',(FNCH))
	data = c.fetchall()
	return data
	
	
def get_NCHC(NCHC):
	c.execute('SELECT * FROM Changements WHERE NCHC=%s',(NCHC))
	data = c.fetchall()
	return data
	
	
def get_FNCHC(FNCHC):
	c.execute('SELECT * FROM Changements WHERE FNCHC=%s',(FNCHC))
	data = c.fetchall()
	return data
	

#========================================

def get_Chantier(Chantier):
	c.execute('SELECT * FROM Anomalies WHERE Chantier=%s',(Chantier))
	data = c.fetchall()
	return data
	

def get_NA(NA):
	c.execute('SELECT * FROM Anomalies WHERE NA=%s',(NA))
	data = c.fetchall()
	return data

	
def get_FNA(FNA):
	c.execute('SELECT * FROM Anomalies WHERE FNA=%s',(FNA))
	data = c.fetchall()
	return data

	
def get_NAC(NAC):
	c.execute('SELECT * FROM Anomalies WHERE NAC=%s',(NAC))
	data = c.fetchall()
	return data

	
def get_FNAC(FNAC):
	c.execute('SELECT * FROM Anomalies WHERE FNAC=%s',(FNAC))
	data = c.fetchall()
	return data
	

	
#==============================

def get_Chantier(Chantier):
	c.execute('SELECT * FROM JSA WHERE Chantier=%s',(Chantier))
	data = c.fetchall()
	return data

	
def get_NAct(NAct):
	c.execute('SELECT * FROM JSA WHERE NAct=%s',(NAct))
	data = c.fetchall()
	return data

	
def get_NJSA(NJSA):
	c.execute('SELECT * FROM JSA WHERE NJSA=%s',(NJSA))
	data = c.fetchall()
	return data
	

#=================================

def get_Chantier(Chantier):
	c.execute('SELECT * FROM Incident_Accident WHERE Chantier=%s',(Chantier))
	data = c.fetchall()
	return data


def get_NInc(NInc):
	c.execute('SELECT * FROM Incident_Accident WHERE NInc=%s',(NInc))
	data = c.fetchall()
	return data

	
def get_AAA(AAA):
	c.execute('SELECT * FROM Incident_Accident WHERE AAA=%s',(AAA))
	data = c.fetchall()
	return data

	
def get_ASA(ASA):
	c.execute('SELECT * FROM Incident_Accident WHERE ASA=%s',(ASA))
	data = c.fetchall()
	return data
	

def get_AT(AT):
	c.execute('SELECT * FROM Incident_Accident WHERE AT=%s',(AT))
	data = c.fetchall()
	return data
	
	
def get_NJP(NJP):
	c.execute('SELECT * FROM Incident_Accident WHERE NJP=%s',(NJP))
	data = c.fetchall()
	return data



	
#==============================

def get_Chantier(Chantier):
	c.execute('SELECT * FROM Audit WHERE Chantier=%s',(Chantier))
	data = c.fetchall()
	return data


def get_AC(AC):
	c.execute('SELECT * FROM Audit WHERE AC=%s',(AC))
	data = c.fetchall()
	return data

	
def get_VC(VC):
	c.execute('SELECT * FROM Audit WHERE VC=%s',(VC))
	data = c.fetchall()
	return data

	
def get_NEU(NEU):
	c.execute('SELECT * FROM Audit WHERE NEU=%s',(NEU))
	data = c.fetchall()
	return data

	
def get_SMPAR(SMPAR):
	c.execute('SELECT * FROM Audit WHERE SMPAR=%s',(SMPAR))
	data = c.fetchall()
	return data

	
def get_NPR(NPR):
	c.execute('SELECT * FROM Audit WHERE NPR=%s',(NPR))
	data = c.fetchall()
	return data

	
def get_IE(IE):
	c.execute('SELECT * FROM Audit WHERE IE=%s',(IE))
	data = c.fetchall()
	return data
	
#====================================
# nous recuperons les email comme ID pour rÃ©fÃ©rencer la base chaque user



def get_IDD(IDD):
	c.execute('SELECT * FROM userstable WHERE email=%s',(email))
	data = c.fetchall()
	return data	


# get_id =================================
def get_id_Accueil(id):
	c.execute('SELECT * FROM Accueil WHERE id=%s;',(id,))
	data = c.fetchall()
	return data	

def get_id_TBM(id):
	c.execute('SELECT * FROM TBM WHERE id=%s;',(id,))
	data = c.fetchall()
	return data	

def get_id_NC(id):
	c.execute('SELECT * FROM NC WHERE id=%s;',(id,))
	data = c.fetchall()
	return data

def get_id_Changements(id):
	c.execute('SELECT * FROM Changements WHERE id=%s;',(id,))
	data = c.fetchall()
	return data	

def get_id_Anomalies(id):
	c.execute('SELECT * FROM Anomalies WHERE id=%s;',(id,))
	data = c.fetchall()
	return data

def get_id_JSA(id):
	c.execute('SELECT * FROM JSA WHERE id=%s;',(id,))
	data = c.fetchall()
	return data

def get_id_Incident_Accident(id):
	c.execute('SELECT * FROM Incident_Accident WHERE id=%s;',(id,))
	data = c.fetchall()
	return data

def get_id_Audit(id):
	c.execute('SELECT * FROM Audit WHERE id=%s;',(id,))
	data = c.fetchall()
	return data

