from flask import Flask, request, jsonify
from pdfminer.high_level import extract_text
import re
import joblib
import fitz
from PIL import Image
import io
import cv2
import numpy as np
import tensorflow as tf
from flask_cors import CORS
import language_tool_python
import os
from spacy.matcher import Matcher
import spacy
from flask import Flask, request, jsonify
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import os
import json
from dotenv import load_dotenv
import re
from collections import Counter
from groq import Groq

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"


app = Flask(__name__)
CORS(app)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
job_profiles_list=[
		"accountant",
		"actor",
		"actuary",
		"adhesive bonding machine tender",
		"adjudicator",
		"administrative assistant",
		"administrative services manager",
		"adult education teacher",
		"advertising manager",
		"advertising sales agent",
		"aerobics instructor",
		"aerospace engineer",
		"aerospace engineering technician",
		"agent",
		"agricultural engineer",
		"agricultural equipment operator",
		"agricultural grader",
		"agricultural inspector",
		"agricultural manager",
		"agricultural sciences teacher",
		"agricultural sorter",
		"agricultural technician",
		"agricultural worker",
		"air conditioning installer",
		"air conditioning mechanic",
		"air traffic controller",
		"aircraft cargo handling supervisor",
		"aircraft mechanic",
		"aircraft service technician",
		"airline copilot",
		"airline pilot",
		"ambulance dispatcher",
		"ambulance driver",
		"amusement machine servicer",
		"anesthesiologist",
		"animal breeder",
		"animal control worker",
		"animal scientist",
		"animal trainer",
		"animator",
		"answering service operator",
		"anthropologist",
		"apparel patternmaker",
		"apparel worker",
		"arbitrator",
		"archeologist",
		"architect",
		"architectural drafter",
		"architectural manager",
		"archivist",
		"art director",
		"art teacher",
		"artist",
		"assembler",
		"astronomer",
		"athlete",
		"athletic trainer",
		"ATM machine repairer",
		"atmospheric scientist",
		"attendant",
		"audio and video equipment technician",
		"audio-visual and multimedia collections specialist",
		"audiologist",
		"auditor",
		"author",
		"auto damage insurance appraiser",
		"automotive and watercraft service attendant",
		"automotive glass installer",
		"automotive mechanic",
		"avionics technician",
    "back-end developer",
		"baggage porter",
		"bailiff",
		"baker",
		"barback",
		"barber",
		"bartender",
		"basic education teacher",
		"behavioral disorder counselor",
		"bellhop",
		"bench carpenter",
		"bicycle repairer",
		"bill and account collector",
		"billing and posting clerk",
		"biochemist",
		"biological technician",
		"biomedical engineer",
		"biophysicist",
		"blaster",
		"blending machine operator",
		"blockmason",
		"boiler operator",
		"boilermaker",
		"bookkeeper",
		"boring machine tool tender",
		"brazer",
		"brickmason",
		"bridge and lock tender",
		"broadcast news analyst",
		"broadcast technician",
		"brokerage clerk",
		"budget analyst",
		"building inspector",
		"bus mechanic",
		"butcher",
		"buyer",
		"cabinetmaker",
		"cafeteria attendant",
		"cafeteria cook",
		"camera operator",
		"camera repairer",
		"cardiovascular technician",
		"cargo agent",
		"carpenter",
		"carpet installer",
		"cartographer",
		"cashier",
		"caster",
		"ceiling tile installer",
		"cellular equipment installer",
		"cement mason",
		"channeling machine operator",
		"chauffeur",
		"checker",
		"chef",
		"chemical engineer",
		"chemical plant operator",
		"chemist",
		"chemistry teacher",
		"chief executive",
		"child social worker",
		"childcare worker",
		"chiropractor",
		"choreographer",
		"civil drafter",
		"civil engineer",
		"civil engineering technician",
		"claims adjuster",
		"claims examiner",
		"claims investigator",
		"cleaner",
		"clinical laboratory technician",
		"clinical laboratory technologist",
		"clinical psychologist",
		"coating worker",
		"coatroom attendant",
		"coil finisher",
		"coil taper",
		"coil winder",
		"coin machine servicer",
		"commercial diver",
		"commercial pilot",
		"commodities sales agent",
		"communications equipment operator",
		"communications teacher",
		"community association manager",
		"community service manager",
		"compensation and benefits manager",
		"compliance officer",
		"composer",
		"computer hardware engineer",
		"computer network architect",
		"computer operator",
		"computer programmer",
		"computer science teacher",
		"computer support specialist",
		"computer systems administrator",
		"computer systems analyst",
		"concierge",
		"conciliator",
		"concrete finisher",
		"conservation science teacher",
		"conservation scientist",
		"conservation worker",
		"conservator",
		"construction inspector",
		"construction manager",
		"construction painter",
		"construction worker",
		"continuous mining machine operator",
		"convention planner",
		"conveyor operator",
		"cook",
		"cooling equipment operator",
		"copy marker",
		"correctional officer",
		"correctional treatment specialist",
		"correspondence clerk",
		"correspondent",
		"cosmetologist",
		"cost estimator",
		"costume attendant",
		"counseling psychologist",
		"counselor",
		"courier",
		"court reporter",
		"craft artist",
		"crane operator",
		"credit analyst",
		"credit checker",
		"credit counselor",
		"criminal investigator",
		"criminal justice teacher",
		"crossing guard",
		"curator",
		"custom sewer",
		"customer service representative",
		"cutter",
		"cutting machine operator",
		"dancer",
		"data entry keyer",
		"database administrator",
		"decorating worker",
		"delivery services driver",
		"demonstrator",
		"dental assistant",
		"dental hygienist",
		"dental laboratory technician",
		"dentist",
    "dermatologist",
		"derrick operator",
		"designer",
		"desktop publisher",
		"detective",
    "developer",
		"diagnostic medical sonographer",
		"die maker",
		"diesel engine specialist",
		"dietetic technician",
		"dietitian",
		"dinkey operator",
		"director",
		"dishwasher",
		"dispatcher",
    "DJ",
    "doctor",
		"door-to-door sales worker",
		"drafter",
		"dragline operator",
		"drama teacher",
		"dredge operator",
		"dressing room attendant",
		"dressmaker",
		"drier operator",
		"drilling machine tool operator",
		"dry-cleaning worker",
		"drywall installer",
		"dyeing machine operator",
		"earth driller",
		"economics teacher",
		"economist",
		"editor",
		"education administrator",
		"electric motor repairer",
		"electrical electronics drafter",
		"electrical engineer",
		"electrical equipment assembler",
		"electrical installer",
		"electrical power-line installer",
		"electrician",
		"electro-mechanical technician",
		"elementary school teacher",
		"elevator installer",
		"elevator repairer",
		"embalmer",
		"emergency management director",
		"emergency medical technician",
		"engine assembler",
		"engineer",
		"engineering manager",
		"engineering teacher",
		"english language teacher",
		"engraver",
		"entertainment attendant",
		"environmental engineer",
		"environmental science teacher",
		"environmental scientist",
		"epidemiologist",
		"escort",
		"etcher",
		"event planner",
		"excavating operator",
		"executive administrative assistant",
		"executive secretary",
		"exhibit designer",
		"expediting clerk",
		"explosives worker",
		"extraction worker",
		"fabric mender",
		"fabric patternmaker",
		"fabricator",
		"faller",
		"family practitioner",
		"family social worker",
		"family therapist",
		"farm advisor",
		"farm equipment mechanic",
		"farm labor contractor",
		"farmer",
		"farmworker",
		"fashion designer",
		"fast food cook",
		"fence erector",
		"fiberglass fabricator",
		"fiberglass laminator",
		"file clerk",
		"filling machine operator",
		"film and video editor",
		"financial analyst",
		"financial examiner",
		"financial manager",
		"financial services sales agent",
		"fine artist",
		"fire alarm system installer",
		"fire dispatcher",
		"fire inspector",
		"fire investigator",
		"firefighter",
		"fish and game warden",
		"fish cutter",
		"fish trimmer",
		"fisher",
		"fitness studies teacher",
		"fitness trainer",
		"flight attendant",
		"floor finisher",
		"floor layer",
		"floor sander",
		"floral designer",
		"food batchmaker",
		"food cooking machine operator",
		"food preparation worker",
		"food science technician",
		"food scientist",
		"food server",
		"food service manager",
		"food technologist",
		"foreign language teacher",
		"foreign literature teacher",
		"forensic science technician",
		"forest fire inspector",
		"forest fire prevention specialist",
		"forest worker",
		"forester",
		"forestry teacher",
		"forging machine setter",
		"foundry coremaker",
		"freight agent",
		"freight mover",
    "front-end developer",
		"fundraising manager",
		"funeral attendant",
		"funeral director",
		"funeral service manager",
		"furnace operator",
		"furnishings worker",
		"furniture finisher",
		"gaming booth cashier",
		"gaming cage worker",
		"gaming change person",
		"gaming dealer",
		"gaming investigator",
		"gaming manager",
		"gaming surveillance officer",
		"garment mender",
		"garment presser",
		"gas compressor",
		"gas plant operator",
		"gas pumping station operator",
		"general manager",
		"general practitioner",
		"geographer",
		"geography teacher",
		"geological engineer",
		"geological technician",
		"geoscientist",
		"glazier",
		"government program eligibility interviewer",
		"graduate teaching assistant",
		"graphic designer",
		"groundskeeper",
		"groundskeeping worker",
		"gynecologist",
		"hairdresser",
		"hairstylist",
		"hand grinding worker",
		"hand laborer",
		"hand packager",
		"hand packer",
		"hand polishing worker",
		"hand sewer",
		"hazardous materials removal worker",
		"head cook",
		"health and safety engineer",
		"health educator",
		"health information technician",
		"health services manager",
		"health specialties teacher",
		"healthcare social worker",
		"hearing officer",
		"heat treating equipment setter",
		"heating installer",
		"heating mechanic",
		"heavy truck driver",
		"highway maintenance worker",
		"historian",
		"history teacher",
		"hoist and winch operator",
		"home appliance repairer",
		"home economics teacher",
		"home entertainment installer",
		"home health aide",
		"home management advisor",
		"host",
		"hostess",
		"hostler",
		"hotel desk clerk",
		"housekeeping cleaner",
		"human resources assistant",
		"human resources manager",
		"human service assistant",
		"hunter",
		"hydrologist",
		"illustrator",
		"industrial designer",
		"industrial engineer",
		"industrial engineering technician",
		"industrial machinery mechanic",
		"industrial production manager",
		"industrial truck operator",
		"industrial-organizational psychologist",
		"information clerk",
		"information research scientist",
		"information security analyst",
		"information systems manager",
		"inspector",
		"instructional coordinator",
		"instructor",
		"insulation worker",
		"insurance claims clerk",
		"insurance sales agent",
		"insurance underwriter",
		"intercity bus driver",
		"interior designer",
		"internist",
		"interpreter",
		"interviewer",
		"investigator",
		"jailer",
		"janitor",
		"jeweler",
		"judge",
		"judicial law clerk",
		"kettle operator",
		"kiln operator",
		"kindergarten teacher",
		"laboratory animal caretaker",
		"landscape architect",
		"landscaping worker",
		"lathe setter",
		"laundry worker",
		"law enforcement teacher",
		"law teacher",
		"lawyer",
		"layout worker",
		"leather worker",
		"legal assistant",
		"legal secretary",
		"legislator",
		"librarian",
		"library assistant",
		"library science teacher",
		"library technician",
		"licensed practical nurse",
		"licensed vocational nurse",
		"life scientist",
		"lifeguard",
		"light truck driver",
		"line installer",
		"literacy teacher",
		"literature teacher",
		"loading machine operator",
		"loan clerk",
		"loan interviewer",
		"loan officer",
		"lobby attendant",
		"locker room attendant",
		"locksmith",
		"locomotive engineer",
		"locomotive firer",
		"lodging manager",
		"log grader",
		"logging equipment operator",
		"logistician",
		"machine feeder",
		"machinist",
		"magistrate judge",
		"magistrate",
		"maid",
		"mail clerk",
		"mail machine operator",
		"mail superintendent",
		"maintenance painter",
		"maintenance worker",
		"makeup artist",
		"management analyst",
		"manicurist",
		"manufactured building installer",
		"mapping technician",
		"marble setter",
		"marine engineer",
		"marine oiler",
		"market research analyst",
		"marketing manager",
		"marketing specialist",
		"marriage therapist",
		"massage therapist",
		"material mover",
		"materials engineer",
		"materials scientist",
		"mathematical science teacher",
		"mathematical technician",
		"mathematician",
		"maxillofacial surgeon",
		"measurer",
		"meat cutter",
		"meat packer",
		"meat trimmer",
		"mechanical door repairer",
		"mechanical drafter",
		"mechanical engineer",
		"mechanical engineering technician",
		"mediator",
		"medical appliance technician",
		"medical assistant",
		"medical equipment preparer",
		"medical equipment repairer",
		"medical laboratory technician",
		"medical laboratory technologist",
		"medical records technician",
		"medical scientist",
		"medical secretary",
		"medical services manager",
		"medical transcriptionist",
		"meeting planner",
		"mental health counselor",
		"mental health social worker",
		"merchandise displayer",
		"messenger",
		"metal caster",
		"metal patternmaker",
		"metal pickling operator",
		"metal pourer",
		"metal worker",
		"metal-refining furnace operator",
		"metal-refining furnace tender",
		"meter reader",
		"microbiologist",
		"middle school teacher",
		"milling machine setter",
		"millwright",
		"mine cutting machine operator",
		"mine shuttle car operator",
		"mining engineer",
		"mining safety engineer",
		"mining safety inspector",
		"mining service unit operator",
		"mixing machine setter",
		"mobile heavy equipment mechanic",
		"mobile home installer",
		"model maker",
		"model",
		"molder",
		"mortician",
		"motel desk clerk",
		"motion picture projectionist",
		"motorboat mechanic",
		"motorboat operator",
		"motorboat service technician",
		"motorcycle mechanic",
    "movers",
		"multimedia artist",
		"museum technician",
		"music director",
		"music teacher",
		"musical instrument repairer",
		"musician",
		"natural sciences manager",
		"naval architect",
		"network systems administrator",
		"new accounts clerk",
		"news vendor",
		"nonfarm animal caretaker",
		"nuclear engineer",
		"nuclear medicine technologist",
		"nuclear power reactor operator",
		"nuclear technician",
		"nursing aide",
		"nursing instructor",
		"nursing teacher",
		"nutritionist",
		"obstetrician",
		"occupational health and safety specialist",
		"occupational health and safety technician",
		"occupational therapist",
		"occupational therapy aide",
		"occupational therapy assistant",
		"offbearer",
		"office clerk",
		"office machine operator",
		"operating engineer",
		"operations manager",
		"operations research analyst",
		"ophthalmic laboratory technician",
		"optician",
		"optometrist",
		"oral surgeon",
		"order clerk",
		"order filler",
		"orderly",
		"ordnance handling expert",
		"orthodontist",
		"orthotist",
		"outdoor power equipment mechanic",
		"oven operator",
		"packaging machine operator",
		"painter ",
		"painting worker",
		"paper goods machine setter",
		"paperhanger",
		"paralegal",
		"paramedic",
		"parking enforcement worker",
		"parking lot attendant",
		"parts salesperson",
		"paving equipment operator",
		"payroll clerk",
		"pediatrician",
		"pedicurist",
		"personal care aide",
		"personal chef",
		"personal financial advisor",
    "personal trainer",
		"pest control worker",
		"pesticide applicator",
		"pesticide handler",
		"pesticide sprayer",
		"petroleum engineer",
		"petroleum gauger",
		"petroleum pump system operator",
		"petroleum refinery operator",
		"petroleum technician",
		"pharmacist",
		"pharmacy aide",
		"pharmacy technician",
		"philosophy teacher",
		"photogrammetrist",
		"photographer",
		"photographic process worker",
		"photographic processing machine operator",
		"physical therapist aide",
		"physical therapist assistant",
		"physical therapist",
		"physician assistant",
		"physician",
		"physicist",
		"physics teacher",
		"pile-driver operator",
		"pipefitter",
		"pipelayer",
		"planing machine operator",
		"planning clerk",
		"plant operator",
		"plant scientist",
		"plasterer",
		"plastic patternmaker",
		"plastic worker",
		"plumber",
		"podiatrist",
		"police dispatcher",
		"police officer",
		"policy processing clerk",
		"political science teacher",
		"political scientist",
		"postal service clerk",
		"postal service mail carrier",
		"postal service mail processing machine operator",
		"postal service mail processor",
		"postal service mail sorter",
		"postmaster",
		"postsecondary teacher",
		"poultry cutter",
		"poultry trimmer",
		"power dispatcher",
		"power distributor",
		"power plant operator",
		"power tool repairer",
		"precious stone worker",
		"precision instrument repairer",
		"prepress technician",
		"preschool teacher",
		"priest",
		"print binding worker",
		"printing press operator",
		"private detective",
		"probation officer",
		"procurement clerk",
		"producer",
		"product promoter",
    "product manager",
		"production clerk",
		"production occupation",
		"proofreader",
		"property manager",
		"prosthetist",
		"prosthodontist",
		"psychiatric aide",
		"psychiatric technician",
		"psychiatrist",
		"psychologist",
		"psychology teacher",
		"public relations manager",
		"public relations specialist",
		"pump operator",
		"purchasing agent",
		"purchasing manager",
		"radiation therapist",
		"radio announcer",
		"radio equipment installer",
		"radio operator",
		"radiologic technician",
		"radiologic technologist",
		"rail car repairer",
		"rail transportation worker",
		"rail yard engineer",
		"rail-track laying equipment operator",
		"railroad brake operator",
		"railroad conductor",
		"railroad police",
		"rancher",
		"real estate appraiser",
		"real estate broker",
		"real estate manager",
		"real estate sales agent",
		"receiving clerk",
		"receptionist",
		"record clerk",
		"recreation teacher",
		"recreation worker",
		"recreational therapist",
		"recreational vehicle service technician",
		"recyclable material collector",
		"referee",
		"refractory materials repairer",
		"refrigeration installer",
		"refrigeration mechanic",
		"refuse collector",
		"regional planner",
		"registered nurse",
		"rehabilitation counselor",
		"reinforcing iron worker",
		"reinforcing rebar worker",
		"religion teacher",
		"religious activities director",
		"religious worker",
		"rental clerk",
		"repair worker",
		"reporter",
		"residential advisor",
		"resort desk clerk",
		"respiratory therapist",
		"respiratory therapy technician",
		"retail buyer",
		"retail salesperson",
		"revenue agent",
		"rigger",
		"rock splitter",
		"rolling machine tender",
		"roof bolter",
		"roofer",
		"rotary drill operator",
		"roustabout",
		"safe repairer",
		"sailor",
		"sales engineer",
		"sales manager",
		"sales representative",
		"sampler",
		"sawing machine operator",
		"scaler",
		"school bus driver",
		"school psychologist",
		"school social worker",
		"scout leader",
		"sculptor",
		"secondary education teacher",
		"secondary school teacher",
		"secretary",
		"securities sales agent",
		"security guard",
		"security system installer",
		"segmental paver",
		"self-enrichment education teacher",
		"semiconductor processor",
		"septic tank servicer",
		"set designer",
		"sewer pipe cleaner",
		"sewing machine operator",
		"shampooer",
		"shaper",
		"sheet metal worker",
		"sheriff's patrol officer",
		"ship captain",
		"ship engineer",
		"ship loader",
		"shipmate",
		"shipping clerk",
		"shoe machine operator",
		"shoe worker",
		"short order cook",
		"signal operator",
		"signal repairer",
		"singer",
		"ski patrol",
		"skincare specialist",
		"slaughterer",
		"slicing machine tender",
		"slot supervisor",
		"social science research assistant",
		"social sciences teacher",
		"social scientist",
		"social service assistant",
		"social service manager",
		"social work teacher",
		"social worker",
		"sociologist",
		"sociology teacher",
		"software developer",
		"software engineer",
		"soil scientist",
		"solderer",
		"sorter",
		"sound engineering technician",
		"space scientist",
		"special education teacher",
		"speech-language pathologist",
		"sports book runner",
		"sports entertainer",
		"sports performer",
		"stationary engineer",
		"statistical assistant",
		"statistician",
		"steamfitter",
		"stock clerk",
		"stock mover",
		"stonemason",
		"street vendor",
		"streetcar operator",
		"structural iron worker",
		"structural metal fabricator",
		"structural metal fitter",
		"structural steel worker",
		"stucco mason",
		"substance abuse counselor",
		"substance abuse social worker",
		"subway operator",
		"surfacing equipment operator",
		"surgeon",
		"surgical technologist",
		"survey researcher",
		"surveying technician",
		"surveyor",
		"switch operator",
		"switchboard operator",
		"tailor",
		"tamping equipment operator",
		"tank car loader",
		"taper",
		"tax collector",
		"tax examiner",
		"tax preparer",
		"taxi driver",
		"teacher assistant",
		"teacher",
		"team assembler",
		"technical writer",
		"telecommunications equipment installer",
		"telemarketer",
		"telephone operator",
		"television announcer",
		"teller",
		"terrazzo finisher",
		"terrazzo worker",
		"tester",
		"textile bleaching operator",
		"textile cutting machine setter",
		"textile knitting machine setter",
		"textile presser",
		"textile worker",
		"therapist",
		"ticket agent",
		"ticket taker",
		"tile setter",
		"timekeeping clerk",
		"timing device assembler",
		"tire builder",
		"tire changer",
		"tire repairer",
		"title abstractor",
		"title examiner",
		"title searcher",
		"tobacco roasting machine operator",
		"tool filer",
		"tool grinder",
		"tool maker",
		"tool sharpener",
		"tour guide",
		"tower equipment installer",
		"tower operator",
		"track switch repairer",
		"tractor operator",
		"tractor-trailer truck driver",
		"traffic clerk",
		"traffic technician",
		"training and development manager",
		"training and development specialist",
		"transit police",
		"translator",
		"transportation equipment painter",
		"transportation inspector",
		"transportation security screener",
		"transportation worker",
		"trapper",
		"travel agent",
		"travel clerk",
		"travel guide",
		"tree pruner",
		"tree trimmer",
		"trimmer",
		"truck loader",
		"truck mechanic",
		"tuner",
		"turning machine tool operator",
    "tutor",
		"typist",
		"umpire",
		"undertaker",
		"upholsterer",
		"urban planner",
		"usher",
    "UX designer",
		"valve installer",
		"vending machine servicer",
		"veterinarian",
		"veterinary assistant",
		"veterinary technician",
		"vocational counselor",
		"vocational education teacher",
		"waiter",
		"waitress",
		"watch repairer",
		"water treatment plant operator",
		"weaving machine setter",
		"web developer",
		"weigher",
		"welder",
		"wellhead pumper",
		"wholesale buyer",
		"wildlife biologist",
		"window trimmer",
		"wood patternmaker",
		"woodworker",
		"word processor",
		"writer",
		"yardmaster",
		"zoologist"
	]
all_skills = [
    "Python", "Java", "C++", "JavaScript", "SQL", "Git", "Problem-solving", "Software architecture",
    "Python", "R", "Machine Learning", "Statistical Analysis", "Data Visualization", "Big Data", "Data Mining", "Predictive Modeling",
    "Cisco Networking", "TCP/IP", "Firewalls", "Routing Protocols", "Network Security", "Troubleshooting", "LAN/WAN", "Wireless Networking",
    "HTML/CSS", "JavaScript", "React", "Node.js", "RESTful APIs", "UI/UX Design", "Responsive Design", "Version Control (Git)",
    "AWS", "Azure", "Google Cloud Platform", "Serverless Architecture", "DevOps", "Containerization (Docker, Kubernetes)", "Infrastructure as Code", "Microservices",
    "SQL", "Database Management", "Performance Tuning", "Data Backup & Recovery", "Database Security", "Data Modeling", "ETL Processes", "NoSQL Databases",
    "Network Security", "Penetration Testing", "Incident Response", "Security Compliance", "Vulnerability Assessment", "SIEM Tools", "Ethical Hacking", "Security Auditing",
    "Machine Learning", "Deep Learning", "Natural Language Processing", "Computer Vision", "TensorFlow", "PyTorch", "Model Deployment", "Algorithm Design",
    "Project Planning", "Agile Methodologies", "Budget Management", "Stakeholder Communication", "Risk Assessment", "Team Leadership", "Resource Allocation", "Change Management",
    "Linux/Unix", "Windows Server", "Active Directory", "System Monitoring", "Backup Solutions", "Virtualization (VMware, Hyper-V)", "Shell Scripting", "Server Configuration",
    "HTML/CSS", "JavaScript", "React", "Angular", "Vue.js", "Responsive Design", "UI/UX Principles", "Cross-browser Compatibility",
    "Python", "Node.js", "Java", "C#", "RESTful APIs", "Database Management", "Microservices Architecture", "Authentication & Authorization",
    "Continuous Integration/Continuous Deployment (CI/CD)", "Containerization (Docker, Kubernetes)", "Infrastructure as Code (Terraform, Ansible)", "Cloud Platforms (AWS, Azure, GCP)", "Monitoring & Logging", "Version Control (Git)", "Scripting (Bash, Python)", "Security Best Practices",
    "User Research", "Wireframing", "Prototyping", "Usability Testing", "Visual Design", "Interaction Design", "Adobe Creative Suite", "Design Thinking",
    "Smart Contracts", "Decentralized Applications (DApps)", "Blockchain Platforms (Ethereum, Hyperledger)", "Cryptocurrencies", "Consensus Algorithms", "Security Auditing", "Tokenization", "Blockchain Development Tools",
    "Troubleshooting", "Hardware Maintenance", "Software Installation", "Remote Desktop Support", "Customer Service", "Ticketing Systems", "Network Configuration", "Technical Documentation",
    "Data Analysis", "Data Visualization Tools (Tableau, Power BI)", "SQL", "ETL Processes", "Business Intelligence Reporting", "Statistical Analysis", "Dashboard Design", "Requirements Gathering",
    "HTML/CSS", "JavaScript", "Python", "Node.js", "React", "Express.js", "Database Management", "RESTful APIs",
    "Test Automation", "Manual Testing", "Test Planning", "Bug Tracking Systems", "Regression Testing", "Performance Testing", "Agile Testing", "Test Case Design",
    "Requirements Gathering", "Business Process Analysis", "Data Modeling", "Use Case Diagrams", "User Acceptance Testing (UAT)", "Agile Methodologies", "Project Documentation", "Stakeholder Communication",
    "Business Analysis", "Project Management", "IT Strategy", "Change Management", "Risk Assessment", "Vendor Management", "Client Relationship Management", "Technical Presentations",
    "iOS Development", "Android Development", "Swift", "Java/Kotlin", "Cross-platform Development (Flutter, React Native)", "Mobile UI/UX Design", "Push Notifications", "App Store Deployment",
    "Programming", "Algorithm Design", "Data Structures", "Version Control (Git)", "Object-Oriented Programming (OOP)", "Testing", "Debugging", "Agile Methodologies",
    "Cybersecurity", "Risk Assessment", "Security Policies", "Penetration Testing", "Incident Response", "Security Awareness Training", "Firewalls", "Encryption",
    "E-commerce Platforms (Magento, Shopify)", "Payment Gateways Integration", "Custom Plugin Development", "User Experience Optimization", "SEO Optimization", "Responsive Design", "Server Management", "API Integration",
    "Game Design", "Unity", "Game Development Frameworks", "3D Modeling", "Game Physics", "Multiplayer Game Development", "Augmented Reality (AR)", "Game Testing",
    "Training Delivery", "Curriculum Development", "Technical Writing", "E-learning Platforms", "Certification Programs", "Instructional Design", "Presentation Skills", "Feedback Analysis",
    "Sales Techniques", "Market Analysis", "Customer Service", "Ms-Excel","Canva designes","Social Media Marketing","Marketing Strategies","Email Marketing","Search Engine Optimization", 
]

tool = language_tool_python.LanguageTool('en-US')
smodel = joblib.load('skillr.pkl')
vectorizer = joblib.load('vectorizer.pkl')
model = tf.keras.models.load_model('newModel.keras')
keywordsModel = joblib.load('keywords.pkl')
keywordsVectorizer = joblib.load('keywordsVectorizer.pkl')
nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)
prompt = ChatPromptTemplate.from_messages([
    ("system", "generate 2 objective of 30 to 40 words to add to my resume, dont add headers to the response"),
    ("user", "Question:{question}")
])
#generate 2 objective of 30 to 40 words to add to my resume, dont add headers to the response
llm = Ollama(model="llama3")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser


def extract_name(resume_text):
    nlp_text = nlp(resume_text)
  
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
    
    matcher.add('NAME', [pattern])
    
    matches = matcher(nlp_text)
    
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        return span.text
    
def check_grammar(text):
 
    matches = tool.check(text)

    
    errors = []
    for match in matches:
      
        error_text = text[match.offset:match.offset + match.errorLength]

        
        if re.search(r'[^\w\s]', error_text):
            continue  

        if not re.match(r'^\s*$', error_text):
            suggestions = [suggestion for suggestion in match.replacements if suggestion.strip()]
            if suggestions:
                errors.append({
                    'text': error_text,  
                    'suggestions': suggestions[:5],  
                })
    return errors


def recommend_skills(job_profile, input_skills):
    input_text = job_profile[0]+ " " + " ".join(input_skills)
    input_vector = vectorizer.transform([input_text])
    
    
    recommended_skills = smodel.predict(input_vector)
    
    return recommended_skills

def recommend_keywords(job_profile, input_skills):
  
    # Vectorize the input job profile and skills
    input_text = job_profile + " " + " ".join(input_skills)
    input_vector = keywordsVectorizer.transform([input_text])

    # Predict additional skills
    recommended_skills = keywordsModel.predict(input_vector)

    return recommended_skills



# Function to extract skills from a resume
def extract_skills_from_resume(text, skills_list):
    skills = []

    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            skills.append(skill)

    return skills

def is_profile_picture(image):
    
    min_width, min_height = 45, 45
    max_width, max_height = 500, 500

    width, height = image.size

  
    return min_width <= width <= max_width and min_height <= height <= max_height


def extract_job_profiles_from_resume(text, job_profiles_list):
    words = re.findall(r'\b\w+\b', text.lower())
    top_100_words = [word for word, _ in Counter(words).most_common(100)]
    
    job_profiles = []

    for job_profile in job_profiles_list:
        pattern = r"\b{}\b".format(re.escape(job_profile))
        # Search only in the top 100 words
        text_to_search = ' '.join(top_100_words)
        match = re.search(pattern, text_to_search, re.IGNORECASE)
        if match:
            job_profiles.append(job_profile)

    return job_profiles
    
    


@app.route('/grammar', methods=['POST'])
def check_grammar_api():
 
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    
    resume_file = request.files['file']
    text = extract_text(resume_file.stream)

    
    errors = check_grammar(text)

    
    return jsonify({'errors': errors})

@app.route('/skills', methods=['POST'])
def extract_and_recommend_skills():
   
    if 'file' not in request.files:
        return jsonify({'error': 'No resume file found in the request.'}), 400
    
    
    resume_file = request.files['file']
    resume_text = extract_text(resume_file.stream)  
  
    

	
    extracted_skills = extract_skills_from_resume(resume_text, all_skills)
    
    Job_profile=extract_job_profiles_from_resume(resume_text, job_profiles_list)
    recommended_skills = recommend_skills(Job_profile, extracted_skills)
    skills_string = recommended_skills[0]

    skills_list = skills_string.split(", ")

    
    
    return jsonify({'extracted_skills': extracted_skills, 'recommended_skills': skills_list})

@app.route('/image', methods=['POST'])
def process_resume():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        profile_picture_path = None 
        try:
            
            resume_path = 'uploaded_resume.pdf'
            file.save(resume_path)

           
            pdf_file = fitz.open(resume_path)

          
            first_page = pdf_file.load_page(0)

           
            image_list = first_page.get_images(full=True)

            
            for img_info in image_list:
               
                xref = img_info[0]

               
                base_image = pdf_file.extract_image(xref)
                image_bytes = base_image["image"]

             
                image_ext = base_image["ext"]

               
                image = Image.open(io.BytesIO(image_bytes))

               
                if is_profile_picture(image):
                  
                    profile_picture_path = f'profile_picture.{image_ext}'
                    image.save(profile_picture_path)
                    print("Profile picture extracted and saved successfully.")
                    break
            else:
                return jsonify({'error': 'No profile picture found in the resume'})

            
            image = cv2.imread(profile_picture_path)
            image = cv2.resize(image, (256, 256))
            preprocessed_image = image.astype('float32') / 255.0

            prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))

            
            result = 'UnProfessional' if prediction[0][0] >= 0.5 else 'Professional'

            
            response = {'prediction': result}

            return jsonify(response)

        except Exception as e:
            print(f"Error: {e}")
            return jsonify({'error': 'An error occurred during processing'})

        finally:
            
            os.remove(resume_path)
            if profile_picture_path is not None: 
                os.remove(profile_picture_path)

    else:
        return jsonify({'error': 'File upload failed'})

    
@app.route('/name', methods=['POST'])
def extract_name_from_resume():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
   
    resume_file = request.files['file']
    text = extract_text(resume_file.stream)
    
    
    name = extract_name(text)
    
    if name:
        return jsonify({'name': name})
    else:
        return jsonify({'error': 'Name not found in the resume'})
    
# @app.route('/api/ask', methods=['POST'])
# def ask():
#     # data = request.json
#     # if 'text' not in data:
#     #     return jsonify({"error": "No text provided."}), 400
#     # print(data)
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided.'}), 400

#     resume_file = request.files['file']

   
#     if resume_file.filename.endswith('.pdf'):
#         # Read the PDF file and extract text
#         text = extract_text(resume_file.stream)


#         extracted_job_profiles = extract_job_profiles_from_resume(text, job_profiles_list)
#     text = extracted_job_profiles[0]
#     print(text)
#     response = chain.invoke({"question": text})
#     print(type(response))
#     print(response)
#     header, *objectives = re.split(r'\n\n|\n\n\*', response)

#     split_objectives = []
#     for objective in objectives:
#         split_lines = objective.strip().split("\n") 
#         bullet_points = [line.strip("* ") for line in split_lines]  
#         split_objectives.append(bullet_points)
#     a = split_objectives[0][0]
#     b = split_objectives[0][1]


#     data = {
#         "header": header.strip(),
#         "obj1": a,
#         "obj2": b
#     }

#     return jsonify(response)

@app.route('/prompt', methods=['POST'])
def prompt():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400

    resume_file = request.files['file']

   
    if resume_file.filename.endswith('.pdf'):
        # Read the PDF file and extract text
        user_content = extract_text(resume_file.stream)

        if not user_content:
            return jsonify({"error": "Failed to extract text from Resume"}), 400
        extracted_job_profiles = extract_job_profiles_from_resume(user_content, job_profiles_list)
        text = extracted_job_profiles[0]
        print(text)
        
        messages = [
            {
                "role": "system",
                "content": "generate an objective of 30 to 40 words to add to my resume, dont add headers to the response"
            },
            {
                "role": "user",
                "content": text
            }
        ]

        # Generate completion
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,  # Set stream to False for simplicity
            stop=None
        )

        # Extract the assistant content from the response
        assistant_content = completion.choices[0].message.content.strip()
        return jsonify({"assistant_content": assistant_content})

@app.route('/job', methods=['POST'])
def extract_job_profiles():
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400

    resume_file = request.files['file']

   
    if resume_file.filename.endswith('.pdf'):
        # Read the PDF file and extract text
        text = extract_text(resume_file.stream)


        extracted_job_profiles = extract_job_profiles_from_resume(text, job_profiles_list)
        return jsonify({'extracted_job_profile': extracted_job_profiles})

    else:
        return jsonify({'error': 'Unsupported file format. Only PDF files are accepted.'}), 400

@app.route('/grade-resume', methods=['POST'])
def grade_resume():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400

    resume_file = request.files['file']

   
    if resume_file.filename.endswith('.pdf'):
        # Read the PDF file and extract text
        user_content = extract_text(resume_file.stream)

        if not user_content:
            return jsonify({"error": "Failed to extract text from Resume"}), 400

        
        messages = [
            {
                "role": "system",
                "content": "You are an strict ATS score expert, you will scan the text and grade it out of 100, reply 1 word answer"
            },
            {
                "role": "user",
                "content": user_content
            }
        ]

        # Generate completion
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,  
            stop=None
        )

        # Extract the assistant content from the response
        assistant_content = completion.choices[0].message.content.strip()

        return jsonify({"ATS": assistant_content})
    
@app.route('/keywords', methods=['POST'])
def keywords():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400

    resume_file = request.files['file']

   
    if resume_file.filename.endswith('.pdf'):
        # Read the PDF file and extract text
        text = extract_text(resume_file.stream)

    skills= extract_skills_from_resume(text, all_skills)    
    profile= extract_job_profiles_from_resume(text, job_profiles_list)
    
    recommend_keywords = recommend_keywords(profile, skills)
    

    return jsonify({"keywords"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
