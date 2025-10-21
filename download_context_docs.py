import os
import json
import time
import re
import random
from datetime import datetime
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException


# --------------------------------------------------------------------------------
# run with:
# python3 download_context_docs.py
# --------------------------------------------------------------------------------

class GoogleScholarPDFDownloader:
    def __init__(self):
        self.output_dir = "xpcs_publications"
        os.makedirs(self.output_dir, exist_ok=True)
        self.download_log = []
        self.setup_driver()
    
    def setup_driver(self):
        """setup Chrome driver with options"""
        chrome_options = Options()
        
        prefs = {
            "download.default_directory": os.path.abspath(self.output_dir),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "plugins.always_open_pdf_externally": True  # download PDFs instead of opening them
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        # commenting out to see the browser actions happen in real time
        # chrome_options.add_argument("--headless")
        
        self.driver = webdriver.Chrome(options=chrome_options)
    
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    def add_delay(self, min_seconds=2, max_seconds=5):
        """add random delay"""
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)
    
    def search_and_download(self, citation, index):
        """search Google Scholar and attempt to download PDF"""
        print(f"\n{'='*60}")
        print(f"Citation #{index}: {citation[:80]}...")
        
        search_url = f"https://scholar.google.com/scholar?q={quote(citation)}"
        print(f"Searching: {search_url}")
        
        try:
            # navigate to Google Scholar
            self.driver.get(search_url)
            self.add_delay(2, 4)
            
            # check if it got blocked
            if "captcha" in self.driver.current_url.lower() or "sorry" in self.driver.title.lower():
                print("Google Scholar is asking for CAPTCHA. Waiting longer...")
                self.add_delay(30, 60)
                return False
            
            # wait for results to load
            wait = WebDriverWait(self.driver, 10)
            
            # look for PDF links
            pdf_found = False
            
            # method 1: look for PDF links - they're usually in a div with class gs_or_ggsm
            try:
                # wait for search results to load
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "gs_r")))
                
                # find PDF links - they're in divs with class gs_or_ggsm
                pdf_elements = self.driver.find_elements(By.CSS_SELECTOR, "div.gs_or_ggsm a")
                
                print(f"Found {len(pdf_elements)} potential PDF links")
                
                for element in pdf_elements:
                    link_text = element.text
                    href = element.get_attribute('href')
                    
                    print(f"  - Link text: '{link_text}', URL: {href[:50]}...")
                    
                    # check if this is a PDF link
                    if '[PDF]' in link_text or href.endswith('.pdf'):
                        print(f"Found PDF link: {href}")
                        
                        # click the link to download
                        print("Clicking PDF link...")
                        element.click()
                        self.add_delay(3, 5)
                        
                        # wait for download to complete
                        time.sleep(5)
                        
                        # check if PDF was downloaded
                        if self.check_download_complete(citation, index):
                            pdf_found = True
                            break
                
                # alternative: try looking for any link with [PDF] text
                if not pdf_found:
                    print("\nTrying alternative PDF link search...")
                    pdf_links = self.driver.find_elements(By.XPATH, "//a[contains(., '[PDF]')]")
                    
                    for link in pdf_links:
                        href = link.get_attribute('href')
                        print(f"Found alternative PDF link: {href}")
                        
                        link.click()
                        self.add_delay(3, 5)
                        time.sleep(5)
                        
                        if self.check_download_complete(citation, index):
                            pdf_found = True
                            break
                
            except Exception as e:
                print(f"Error finding PDF links: {e}")
                
                # debug: print page source snippet to see what's there
                try:
                    results = self.driver.find_elements(By.CLASS_NAME, "gs_r")
                    if results:
                        print("\nDebug - First result HTML snippet:")
                        print(results[0].get_attribute('innerHTML')[:500])
                except:
                    pass
            
            if pdf_found:
                self.download_log.append({
                    'status': 'success',
                    'citation': citation,
                    'index': index
                })
                return True
            else:
                print("Could not find or download PDF")
                self.download_log.append({
                    'status': 'failed',
                    'citation': citation,
                    'reason': 'No PDF found or download failed'
                })
                
                # save the page URL for manual access
                self.save_manual_url(citation, index, self.driver.current_url)
                return False
                
        except Exception as e:
            print(f"Error during search: {e}")
            self.download_log.append({
                'status': 'error',
                'citation': citation,
                'error': str(e)
            })
            return False
    
    def check_download_complete(self, citation, index):
        """check if a PDF was downloaded and rename it"""
        try:
            # extract filename info
            year_match = re.search(r'\b(19|20)\d{2}\b', citation)
            year = year_match.group() if year_match else ""
            author_match = re.match(r'^([A-Za-z]+)', citation)
            author = author_match.group(1) if author_match else ""
            
            filename = f"{index:03d}_{author}_{year}.pdf"
            
            # find the most recent PDF in the download directory
            pdf_files = [f for f in os.listdir(self.output_dir) if f.endswith('.pdf')]
            
            if pdf_files:
                # get the most recent file
                latest_pdf = max(pdf_files, key=lambda f: os.path.getctime(os.path.join(self.output_dir, f)))
                latest_path = os.path.join(self.output_dir, latest_pdf)
                
                # check if this file was created in the last 10 seconds
                if time.time() - os.path.getctime(latest_path) < 10:
                    # rename to our naming convention
                    new_path = os.path.join(self.output_dir, filename)
                    if latest_path != new_path:
                        os.rename(latest_path, new_path)
                        print(f"Downloaded and renamed to: {filename}")
                    else:
                        print(f"Downloaded: {filename}")
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error checking download: {e}")
            return False
    
    def save_manual_url(self, citation, index, url):
        """save URL for manual download"""
        urls_file = os.path.join(self.output_dir, f"{index:03d}_manual_download.txt")
        with open(urls_file, 'w') as f:
            f.write(f"Citation: {citation}\n\n")
            f.write(f"Google Scholar URL: {url}\n\n")
            f.write("This paper needs to be downloaded manually.\n")
    
    def process_citations(self, citations, start_index=1):
        """process all citations with added delays"""
        print("AUTOMATED GOOGLE SCHOLAR PDF DOWNLOADER")
        print("======================================")
        print(f"Will process {len(citations)} citations starting from #{start_index}")
        
        success_count = 0
        
        for i, citation in enumerate(citations):
            actual_index = start_index + i  # calculate the actual citation number
            if self.search_and_download(citation.strip(), actual_index):
                success_count += 1
            
            # longer delay between searches to avoid blocking
            if i < len(citations) - 1:  # don't wait after the last citation
                delay = random.uniform(10, 20)
                print(f"\nWaiting {delay:.1f} seconds before next search...")
                time.sleep(delay)
        
        # save download log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.output_dir, f"download_log_batch_{start_index}_{timestamp}.json")
        with open(log_file, 'w') as f:
            json.dump(self.download_log, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Summary: Downloaded {success_count}/{len(citations)} papers")
        print(f"Check {self.output_dir}/ for PDFs")
        print(f"Log saved as: {log_file}")
        
        # close driver
        self.driver.quit()

# citations present in the 2018 review
# citations = [
#     "Livet F. 2007. Diffraction with a coherent X-ray beam: dynamics and imaging. Acta Crystallogr. A 63:87–107",
#     "Grübel G, Madsen A, Robert A. 2008. X-ray photon correlation spectroscopy (XPCS). In Soft-Matter Characterization, ed. R Borsali, R Pecora, pp. 953–95. New York: Springer",
#     "Sutton M. 2008. A review of X-ray intensity fluctuation spectroscopy. C. R. Phys. 9:657–67",
# ]

# all 115 citations present in the 2018 review
citations = [
    "Livet F. 2007. Diffraction with a coherent X-ray beam: dynamics and imaging. Acta Crystallogr. A 63:87–107",
    "Grübel G, Madsen A, Robert A. 2008. X-ray photon correlation spectroscopy (XPCS). In Soft-Matter Characterization, ed. R Borsali, R Pecora, pp. 953–95. New York: Springer",
    "Sutton M. 2008. A review of X-ray intensity fluctuation spectroscopy. C. R. Phys. 9:657–67",
    "Livet F, Sutton M. 2012. X-ray coherent scattering in metal physics. C. R. Phys. 13:23–32",
    "Sinha SK, Jiang Z, Lurio LB. 2014. X-ray photon correlation spectroscopy studies of surfaces and thin films. Adv. Mater. 26:7764–85",
    "Madsen A, Fluerasu A, Ruta B. 2016. Structural dynamics of materials probed by X-ray photon correlation spectroscopy. In Synchrotron Light Sources and Free-Electron Lasers, ed. E Jaeschke, S Khan, JR Schneider, JB Hastings, pp. 1617–41. Cham, Switz.: Springer Int.",
    "Sutton M, Mochrie SGJ, Greytak T, Nagler SE, Berman LE, et al. 1991. Observation of speckle by diffraction with coherent X-rays. Nature 352:608–10",
    "Miao J, Ishikawa T, Robinson IK, Murnane MM. 2015. Beyond crystallography: diffractive imaging using coherent X-ray light sources. Science 348(6234):530–35",
    "Born M, Wolf E. 1980. Principles of Optics: Electromagnetic Theory of Propagation, Interference and Diffraction of Light. Oxford, UK: Pergamon",
    "Goodman JW. 1985. Statistical Optics. New York: Wiley",
    "Vartanyants IA, Singer A. 2010. Coherence properties of hard X-ray synchrotron sources and X-ray free electron lasers. New J. Phys. 12:035004",
    "Sandy AR, Lurio LB, Mochrie SGJ, Malik A, Stephenson GB, et al. 1999. Design and characterization of an undulator beamline optimized for small-angle coherent X-ray scattering at the Advanced Photon Source. J. Synchrotron Radiat. 6:1174–84",
    "Pfeiffer F, Zhang W, Robinson IK. 2004. Coherent grazing exit X-ray scattering geometry for probing the structure of thin films. Appl. Phys. Lett. 84:1847–49",
    "Markowitz D, Kadanoff LP. 1963. Effect of impurities upon critical temperature of anisotropic superconductors. Phys. Rev. 131:563–75",
    "Dierker SB, Pindak R, Fleming RM, Robinson IK, Berman L. 1995. X-ray photon-correlation spectroscopy study of Brownian motion of gold colloids in glycerol. Phys. Rev. Lett. 75:449–52",
    "Zhang F, Allen AJ, Levine LE, Espinal L, Antonucci JM, et al. 2012. Ultra-small-angle X-ray scattering–X-ray photon correlation spectroscopy studies of incipient structural changes in amorphous calcium phosphate–based dental composites. J. Biomed. Mater. Res. A 100:1293–306",
    "Zhang Q, Dufresne EM, Chen P, Park J, Cosgriff MP, et al. 2017. Thermal fluctuations of ferroelectric nanodomains in a ferroelectric-dielectric PbTiO3/SrTiO3 superlattice. Phys. Rev. Lett. 118:097601",
    "Barbour A, Alatas A, Liu Y, Zhu C, Leu BM, et al. 2016. Partial glass isosymmetry transition in multiferroic hexagonal ErMnO3. Phys. Rev. B 93:054113",
    "Singer A, Patel SKK, Uhlíř V, Kukreja R, Ulvestad A, et al. 2016. Phase coexistence and pinning of charge density waves by interfaces in chromium. Phys. Rev. B 94:174110",
    "Shpyrko OG, Isaacs ED, Logan JM, Feng YJ, Aeppli G, et al. 2007. Direct measurement of antiferromagnetic domain fluctuations. Nature 447:68–71",
    "Su J, Sandy AR, Mohanty J, Shpyrko OG, Sutton M. 2012. Collective pinning dynamics of charge-density waves in 1T-TaS2. Phys. Rev. B 86:205105",
    "Fluerasu A, Sutton M, Dufresne EM. 2005. X-ray intensity fluctuation spectroscopy studies on phase-ordering systems. Phys. Rev. Lett. 94:055501",
    "Sanborn C, Ludwig KF, Rogers MC, Sutton M. 2011. Direct measurement of microstructural avalanches during the martensitic transition of cobalt using coherent X-ray scattering. Phys. Rev. Lett. 107:015702",
    "Livet F, Fèvre M, Beutier G, Sutton M. 2015. Ordering fluctuation dynamics in AuAgZn2. Phys. Rev. B 92:094102",
    "Pierce MS, Chang KC, Hennessy D, Komanicky V, Sprung M, et al. 2009. Surface X-ray speckles: coherent surface diffraction from Au(001). Phys. Rev. Lett. 103:165501",
    "Pierce MS, Hennessy DC, Chang KC, Komanicky V, Strzalka J, et al. 2011. Persistent oscillations of X-ray speckles: Pt(001) step flow. Appl. Phys. Lett. 99:121910",
    "Pierce MS, Komanicky V, Barbour A, Hennessy DC, Su JD, et al. 2011. In-situ coherent X-ray scattering and scanning tunneling microscopy studies of hexagonally reconstructed Au(001) in electrolytes. ECS Trans. 35:71–81",
    "Pierce MS, Komanicky V, Barbour A, Hennessy DC, Zhu CH, et al. 2012. Dynamics of the Au(001) surface in electrolytes: in situ coherent X-ray scattering. Phys. Rev. B 86:085410",
    "Pierce MS, Barbour A, Komanicky V, Hennessy D, You H. 2012. Coherent X-ray scattering experiments of Pt(001) surface dynamics near a roughening transition. Phys. Rev. B 86:184108",
    "Karl RM, Barbour A, Komanicky V, Zhu CH, Sandy A, et al. 2015. Charge-induced equilibrium dynamics and structure at the Ag(001)-electrolyte interface. Phys. Chem. Chem. Phys. 17:16682–87",
    "Narayanan RA, Thiyagarajan P, Lewis S, Bansal A, Schadler LS, Lurio LB. 2006. Dynamics and internal stress at the nanoscale related to unique thermomechanical behavior in polymer nanocomposites. Phys. Rev. Lett. 97:075505",
    "Narayanan S, Lee DR, Hagman A, Li XF, Wang J. 2007. Particle dynamics in polymer-metal nanocomposite thin films on nanometer-length scales. Phys. Rev. Lett. 98:185506",
    "Kandar AK, Srivastava S, Basu JK, Mukhopadhyay MK, Seifert S, Narayanan S. 2009. Unusual dynamical arrest in polymer grafted nanoparticles. J. Chem. Phys. 130:121102",
    "Srivastava S, Kandar AK, Basu JK, Mukhopadhyay MK, Lurio LB, et al. 2009. Complex dynamics in polymer nanocomposites. Phys. Rev. E 79:021408",
    "Srivastava S, Chandran S, Kandar AK, Sarika CK, Basu JK, et al. 2010. Communication: Unusual dynamics of hybrid nanoparticles and their binary mixtures. J. Chem. Phys. 133:151105",
    "Sikorski M, Sandy AR, Narayanan S. 2011. Depletion-induced structure and dynamics in bimodal colloidal suspensions. Phys. Rev. Lett. 106:188301",
    "Guo H, Bourret G, Lennox RB, Sutton M, Harden JL, Leheny RL. 2012. Entanglement-controlled subdiffusion of nanoparticles within concentrated polymer solutions. Phys. Rev. Lett. 109:055901",
    "Kim D, Srivastava S, Narayanan S, Archer LA. 2012. Polymer nanocomposites: polymer and particle dynamics. Soft Matter 8:10813–18",
    "Srivastava S, Archer LA, Narayanan S. 2013. Structure and transport anomalies in soft colloids. Phys. Rev. Lett. 110:148302",
    "Jang WS, Koo P, Bryson K, Narayanan S, Sandy A, et al. 2014. Dynamics of cadmium sulfide nanoparticles within polystyrene melts. Macromolecules 47:6483–90",
    "Agrawal A, Yu H-Y, Srivastava S, Choudhury S, Narayanan S, Archer LA. 2015. Dynamics and yielding of binary self-suspended nanoparticle fluids. Soft Matter 11:5224–34",
    "Ranka M, Varkey N, Ramakrishnan S, Zukoski CF. 2015. Impact of small changes in particle surface chemistry for unentangled polymer nanocomposites. Soft Matter 11:1634–45",
    "Srivastava S, Agarwal P, Mangal R, Koch DL, Narayanan S, Archer LA. 2015. Hyperdiffusive dynamics in Newtonian nanoparticle fluids. ACS Macro Lett. 4:1149–53",
    "Grein-Iankovski A, Riegel-Vidotti IC, Simas-Tosin FF, Narayanan S, Leheny RL, Sandy AR. 2016. Exploring the relationship between nanoscale dynamics and macroscopic rheology in natural polymer gums. Soft Matter 12:9321–29",
    "Jang WS, Koo P, Bryson K, Narayanan S, Sandy AR, et al. 2016. The static structure and dynamics of cadmium sulfide nanoparticles within poly(styrene-block-isoprene) diblock copolymer melts. Macromol. Chem. Phys. 217:591–98",
    "Liu SQ, Senses E, Jiao Y, Narayanan S, Akcora P. 2016. Structure and entanglement factors on dynamics of polymer-grafted nanoparticles. ACS Macro Lett. 5:569–73",
    "Mangal R, Srivastava S, Narayanan S, Archer LA. 2016. Size-dependent particle dynamics in entangled polymer nanocomposites. Langmuir 32:596–603",
    "Poling-Skutvik R, Mongcopa KIS, Faraone A, Narayanan S, Conrad JC, Krishnamoorti R. 2016. Structure and dynamics of interacting nanoparticles in semidilute polymer solutions. Macromolecules 49:6568–77",
    "Srivastava S, Kishore S, Narayanan S, Sandy AR, Bhatia SR. 2016. Multiple dynamic regimes in colloid-polymer dispersions: new insight using X-ray photon correlation spectroscopy. J. Polym. Sci. B Polym. Phys. 54:752–60",
    "Lee J, Grein-Iankovski A, Narayanan S, Leheny RL. 2017. Nanorod mobility within entangled wormlike micelle solutions. Macromolecules 50:406–15",
    "Senses E, Ansar SM, Kitchens CL, Mao Y, Narayanan S, et al. 2017. Small particle driven chain disentanglements in polymer nanocomposites. Phys. Rev. Lett. 118:147801",
    "Broennimann C, Eikenberry EF, Henrich B, Horisberger R, Huelsen G, et al. 2006. The PILATUS 1M detector. J. Synchrotron Radiat. 13:120–30",
    "Johnson I, Bergamaschi A, Buitenhuis J, Dinapoli R, Greiffenberg D, et al. 2012. Capturing dynamics with Eiger, a fast-framing X-ray detector. J. Synchrotron Radiat. 19:1001–5",
    "Pennicard D, Lange S, Smoljanin S, Hirsemann H, Graafsma H. 2012. LAMBDA—Large Area Medipix3-Based Detector Array. J. Instrum. 7:C11009",
    "Chen M. 2011. A brief overview of bulk metallic glasses. NPG Asia Mater. 3:82–90",
    "Schroers J. 2013. Bulk metallic glasses. Phys. Today 66:32–37",
    "Byrne CJ, Eldrup M. 2008. Bulk metallic glasses. Science 321:502–3",
    "Martinez L-M, Angell CA. 2001. A thermodynamic connection to the fragility of glass-forming liquids. Nature 410:663–67",
    "Berthier L, Biroli G, Bouchaud J-P, Cipelletti L, Masri DE, et al. 2005. Direct experimental evidence of a growing length scale accompanying the glass transition. Science 310:1797–800",
    "Ruta B, Baldi G, Chushkin Y, Ruffle B, Cristofolini L, et al. 2014. Revealing the fast atomic motion of network glasses. Nat. Commun. 5:3939",
    "Giordano VM, Ruta B. 2016. Unveiling the structural arrangements responsible for the atomic dynamics in metallic glasses during physical aging. Nat. Commun. 7:10344",
    "Ruta B, Chushkin Y, Monaco G, Cipelletti L, Pineda E, et al. 2012. Atomic-scale relaxation dynamics and aging in a metallic glass probed by X-ray photon correlation spectroscopy. Phys. Rev. Lett. 109:165701",
    "Evenson Z, Ruta B, Hechler S, Stolpe M, Pineda E, et al. 2015. X-ray photon correlation spectroscopy reveals intermittent aging dynamics in a metallic glass. Phys. Rev. Lett. 115:175701",
    "Thurn-Albrecht T, Steffen W, Patkowski A, Meier G, Fischer EW, et al. 1996. Photon correlation spectroscopy of colloidal palladium using a coherent X-ray beam. Phys. Rev. Lett. 77:5437–40",
    "Lal J, Abernathy D, Auvray L, Diat O, Grübel G. 2001. Dynamics and correlations in magnetic colloidal systems studied by X-ray photon correlation spectroscopy. Eur. Phys. J. E 4:263–71",
    "Autenrieth T, Robert A, Wagner J, Grübel G. 2007. The dynamic behavior of magnetic colloids in suspension. J. Appl. Crystallogr. 40:S250–53",
    "Fluerasu A, Moussaïd A, Madsen A, Schofield A. 2007. Slow dynamics and aging in colloidal gels studied by X-ray photon correlation spectroscopy. Phys. Rev. E 76:010401(R)",
    "Trappe V, Pitard E, Ramos L, Robert A, Bissig H, Cipelletti L. 2007. Investigation of q-dependent dynamical heterogeneity in a colloidal gel by X-ray photon correlation spectroscopy. Phys. Rev. E 76:051404",
    "Robert A, Wagner J, Haertl W, Autenrieth T, Grübel G. 2008. Dynamics in dense suspensions of charge-stabilized colloidal particles. Eur. Phys. J. E 25:77–81",
    "Guo HY, Ramakrishnan S, Harden JL, Leheny RL. 2010. Connecting nanoscale motion and rheology of gel-forming colloidal suspensions. Phys. Rev. E 81:050401(R)",
    "Guo H, Ramakrishnan S, Harden JL, Leheny RL. 2011. Gel formation and aging in weakly attractive nanocolloid suspensions at intermediate concentrations. J. Chem. Phys. 135:154903",
    "Spannuth M, Mochrie SGJ, Peppin SSL, Wettlaufer JS. 2011. Dynamics of colloidal particles in ice. J. Chem. Phys. 135:224706",
    "Orsi D, Fluerasu A, Moussaid A, Zontone F, Cristofolini L, Madsen A. 2012. Dynamics in dense hard-sphere colloidal suspensions. Phys. Rev. E 85:011402",
    "Westermeier F, Fischer B, Roseker W, Grübel G, Naegele G, Heinen M. 2012. Structure and short-time dynamics in concentrated suspensions of charged colloids. J. Chem. Phys. 137:114504",
    "Angelini R, Zulian L, Fluerasu A, Madsen A, Ruocco G, Ruzicka B. 2013. Dichotomic aging behaviour in a colloidal glass. Soft Matter 9:10955–59",
    "Zhang F, Allen AJ, Levine LE, Ilavsky J, Long GG. 2013. Structure and dynamics studies of concentrated micrometer-sized colloidal suspensions. Langmuir 29:1379–87",
    "Angelini R, Madsen A, Fluerasu A, Ruocco G, Ruzicka B. 2014. Aging behavior of the localization length in a colloidal glass. Colloids Surf. A Physicochem. Eng. Asp. 460:118–22",
    "Angelini R, Zaccarelli E, Marques FAD, Sztucki M, Fluerasu A, et al. 2014. Glass-glass transition during aging of a colloidal clay. Nat. Commun. 5:4049",
    "Marques FAD, Angelini R, Zaccarelli E, Farago B, Ruta B, et al. 2015. Structural and microscopic relaxations in a colloidal glass. Soft Matter 11:466–71",
    "Pusey PN. 1991. Colloidal suspensions. In Liquids, Freezing and the Glass Transition, ed. JP Hansen, D Levesque, J Zinn-Justin, pp. 763–942. Amsterdam: Elsevier",
    "de Gennes P-G. 1979. Scaling Concepts in Polymer Physics. Ithaca, NY: Cornell Univ. Press",
    "Hunter GL, Weeks ER. 2012. The physics of the colloidal glass transition. Rep. Prog. Phys. 75:066501",
    "Lumma D, Lurio LB, Borthwick MA, Falus P, Mochrie SGJ. 2000. Structure and dynamics of concentrated dispersions of polystyrene latex spheres in glycerol: static and dynamic X-ray scattering. Phys. Rev. E 62:8258–69",
    "Beysens D, Narayanan T. 1999. Wetting-induced aggregation of colloids. J. Stat. Phys. 95:997–1008",
    "Pontoni D, Narayanan T, Petit JM, Grübel G, Beysens D. 2003. Microstructure and dynamics near an attractive colloidal glass transition. Phys. Rev. Lett. 90:188301",
    "Lu XH, Mochrie SGJ, Narayanan S, Sandy AR, Sprung M. 2008. How a liquid becomes a glass both on cooling and on heating. Phys. Rev. Lett. 100:045701",
    "Lu XH, Mochrie SGJ, Narayanan S, Sandy AR, Sprung M. 2010. Temperature-dependent structural arrest of silica colloids in a water-lutidine binary mixture. Soft Matter 6:6160–77",
    "Falus P, Borthwick MA, Mochrie SGJ. 2004. Fast CCD camera for X-ray photon correlation spectroscopy and time-resolved X-ray scattering and imaging. Rev. Sci. Instrum. 75:4383–400",
    "Götze W, Sperl M. 2002. Logarithmic relaxation in glass-forming systems. Phys. Rev. E 66:011405",
    "Cipelletti L, Manley S, Ball RC, Weitz DA. 2000. Universal aging features in the restructuring of fractal colloidal gels. Phys. Rev. Lett. 84(10):2275–78",
    "Bouchaud JP, Pitard E. 2001. Anomalous dynamical light scattering in soft glassy gels. Eur. Phys. J. E 6:231–36",
    "Akcora P, Kumar SK, Moll J, Lewis S, Schadler LS, et al. 2010. \"Gel-like\" mechanical reinforcement in polymer nanocomposite melts. Macromolecules 43:1003–10",
    "Chen XM, Thampy V, Mazzoli C, Barbour AM, Miao H, et al. 2016. Remarkable stability of charge density wave order in La1.875Ba0.125CuO4. Phys. Rev. Lett. 117:167001",
    "Evans PG, Isaacs ED, Aeppli G, Cai Z, Lai B. 2002. X-ray microdiffraction images of antiferromagnetic domain evolution in chromium. Science 295(5557):1042–45",
    "Tripathi A, Mohanty J, Dietze SH, Shpyrko OG, Shipton E, et al. 2011. Dichroic coherent diffractive imaging. PNAS 108:13393–98",
    "Parkin SS, Hayashi M, Thomas L. 2008. Magnetic domain-wall racetrack memory. Science 320:190–94",
    "Allwood DA, Xiong G, Faulkner CC, Atkinson D, Petit D, Cowburn RP. 2005. Magnetic domain-wall logic. Science 309:1688–92",
    "Malik A, Sandy AR, Lurio LB, Stephenson GB, Mochrie SGJ, et al. 1998. Coherent X-ray study of fluctuations during domain coarsening. Phys. Rev. Lett. 81:5832–35",
    "Bikondoa O. 2017. On the use of two-time correlation functions for X-ray photon correlation spectroscopy data analysis. J. Appl. Crystallogr. 50:357–68",
    "Bonn D, Kellay H, Tanaka H, Wegdam G, Meunier J. 1999. Laponite: What is the difference between a gel and a glass? Langmuir 15(22):7534–36",
    "Zaccarelli E. 2007. Colloidal gels: equilibrium and non-equilibrium routes. J. Phys. Condens. Matter 19:323101",
    "Zhang Q, Dufresne EM, Grybos P, Kmon P, Maj P, et al. 2016. Submillisecond X-ray photon correlation spectroscopy from a pixel array detector with fast dual gating and no readout dead-time. J. Synchrotron Radiat. 23:679–84",
    "Grybos P, Kmon P, Maj P, Szczygiel R. 2016. 32k channel readout IC for single photon counting pixel detectors with 75 µm pitch, dead time of 85 ns, 9 e− offset spread and 2% rms gain spread. IEEE Trans. Nucl. Sci. 63:1155–61",
    "Zhang Q, Bahadur D, Dufresne EM, Grybos P, Kmon P, et al. 2017. Dynamic scaling of colloidal gel formation at intermediate concentrations. Phys. Rev. Lett. 119:178006",
    "Ramakrishnan S, Chen YL, Schweizer KS, Zukoski CF. 2004. Elasticity and clustering in concentrated depletion gels. Phys. Rev. E 70:040401",
    "Zia RN, Landrum BJ, Russel WB. 2014. A micro-mechanical study of coarsening and rheology of colloidal gels: cage building, cage hopping, and Smoluchowski's ratchet. J. Rheol. 58(5):1121–57",
    "Varga Z, Wang G, Swan J. 2015. The hydrodynamics of colloidal gelation. Soft Matter 11(46):9009–19",
    "Ulvestad A, Singer A, Clark JN, Cho HM, Kim JW, et al. 2015. Topological defect dynamics in operando battery nanoparticles. Science 348:1344–47",
    "Pierce MS, Moore RG, Sorensen LB, Kevan SD, Hellwig O, et al. 2003. Quasistatic X-ray speckle metrology of microscopic magnetic return-point memory. Phys. Rev. Lett. 90:175502",
    "Rogers MC, Chen K, Andrzejewski L, Narayanan S, Ramakrishnan S, et al. 2014. Echoes in X-ray speckles track nanometer-scale plastic events in colloidal gels under shear. Phys. Rev. E 90:062310",
    "Ruta B, Baldi G, Monaco G, Chushkin Y. 2013. Compressed correlation functions and fast aging dynamics in metallic glasses. J. Chem. Phys. 138:054508",
    "Evenson Z, Payes-Playa A, Chushkin Y, di Michiel M, Pineda E, Ruta B. 2017. Comparing the atomic and macroscopic aging dynamics in an amorphous and partially crystalline Zr44Ti11Ni10Cu10Be25 bulk metallic glass. J. Mater. Res. 32:2014–21",
    "Eriksson M, van der Veen JF, Quitmann C. 2014. Diffraction-limited storage rings—a window to the science of tomorrow. J. Synchrotron Radiat. 21:837–42",
    "Hettel R. 2014. DLSR design and plans: an international overview. J. Synchrotron Radiat. 21:843–55",
    "Jakeman E. 1973. Photon correlation. In Photon Correlation and Light Beating Spectroscopy, ed. HZ Cummins, ER Pike, pp. 75–149. New York: Plenum"
]

if __name__ == "__main__":

    #print(f"len of citations: {len(citations)}")

    #batch_citations = citations[3:20]
    #batch_citations = citations[20:40] # get docs 21 - 40
    #batch_citations = citations[36:40] # get docs 37 - 40
    #batch_citations = citations[40:60] # get docs 41 - 60
    #batch_citations = citations[60:80] # get docs 61 - 80
    #batch_citations = citations[80:100] # get docs 81 - 100

    batch_citations = citations[100:115] # get docs 101 - 115

    print(f"Processing citations 101-115 ({len(batch_citations)} total)")

    downloader = GoogleScholarPDFDownloader()
    downloader.process_citations(batch_citations, start_index=101)

    print("\n\n Completed downloads. \n\n")
    
    print("\n" + "="*60)
    print("Batch complete!")
    #print("\nNext batch would be citations 101-115:")
