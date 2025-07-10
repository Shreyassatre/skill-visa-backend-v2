# -*- coding: utf-8 -*-
import copy
import logging
import os
import json
import tempfile
from typing import Optional, Dict, Any, List
from bson import ObjectId
import certifi
import mammoth
import uvicorn
import PyPDF2
from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, File, Path, Query, UploadFile, HTTPException, Form, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from pydantic import BaseModel, EmailStr, Field, ValidationError
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import ConnectionFailure, OperationFailure
import numpy as np
import requests
from datetime import datetime, timezone

from auth_utils import get_current_user, verify_access
from db import close_db_connection, initialize_db
from google_auth import router as google_auth_router
from auth_endpoints import router as auth_router
from starlette.middleware.sessions import SessionMiddleware

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = "resume_parser_db"
RESUME_COLLECTION_NAME = os.getenv("MONGO_RESUME_COLLECTION_NAME", "uploaders_resumes")
OCCUPATION_COLLECTION_NAME = os.getenv("MONGO_OCCUPATION_COLLECTION_NAME", "occupation_suggestions")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MAX_FILE_SIZE = 10 * 1024 * 1024
OCCUPATIONS_EMBEDDINGS_FILE = "aus_visa_occupations_with_embeddings.json"
SIMILARITY_THRESHOLD = 0.7

if not MONGO_URI:
    logger.error("MONGODB_URI environment variable not set")
    raise ValueError("MONGODB_URI environment variable not set")
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY environment variable not set")
    raise ValueError("GOOGLE_API_KEY environment variable not set")

try:
    ca = certifi.where()
    mongo_client = MongoClient(MONGO_URI, tlsCAFile=ca, serverSelectionTimeoutMS=10000)
    mongo_client.admin.command('ping')
    db = mongo_client[DB_NAME]
    resume_collection = db[RESUME_COLLECTION_NAME]
    occupation_collection = db[OCCUPATION_COLLECTION_NAME]
    resume_collection.create_index("candidates.candidate_email")
    occupation_collection.create_index([("uploader_email", 1), ("candidate_email", 1)], unique=True)
    logger.info(f"Connected to MongoDB. DB: {DB_NAME}, Collections: {RESUME_COLLECTION_NAME}, {OCCUPATION_COLLECTION_NAME}")
except (ConnectionFailure, OperationFailure, Exception) as e:
    logger.exception(f"MongoDB setup failed: {e}")
    raise

try:
    gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
    gemini_model_name = "gemini-2.0-flash-001"
    logger.info(f"Gemini API configured successfully for model: {gemini_model_name}")
except Exception as e:
    logger.exception(f"Error configuring Gemini API: {e}")
    raise

occupation_data = []
try:
    with open(OCCUPATIONS_EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
        for item in all_data:
            if (item.get("Occupation") and
                item.get("ANZSCO") and
                isinstance(item.get("occupation_embedding"), list) and
                len(item["occupation_embedding"]) > 0):
                occupation_data.append({
                    "ANZSCO": item["ANZSCO"],
                    "Occupation": item["Occupation"],
                    "embedding": item["occupation_embedding"]
                })
        logger.info(f"Loaded and processed {len(occupation_data)} occupations with embeddings from {OCCUPATIONS_EMBEDDINGS_FILE}.")
        if not occupation_data:
            logger.warning(f"Warning: No valid occupations with embeddings found in {OCCUPATIONS_EMBEDDINGS_FILE}. Matching may be limited.")
except FileNotFoundError:
    logger.error(f"Error: Occupations file '{OCCUPATIONS_EMBEDDINGS_FILE}' not found. Occupation matching will be disabled.")
except json.JSONDecodeError:
    logger.error(f"Error: Could not decode JSON from '{OCCUPATIONS_EMBEDDINGS_FILE}'. Occupation matching will be disabled.")
except Exception as e:
    logger.exception(f"Error loading or processing occupation data from {OCCUPATIONS_EMBEDDINGS_FILE}: {e}")


app = FastAPI(title="AI Resume Parser & Occupation Suggester",
              description="Extracts data from resumes using Gemini, stores in MongoDB, suggests occupations, matches to ANZSCO codes using embeddings, and stores suggestions.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add session middleware for OAuth
app.add_middleware(
    SessionMiddleware,
    secret_key="YOUR_SESSION_SECRET_KEY"  # Replace with a secure secret key
)

# Include the authentication routers
app.include_router(google_auth_router, tags=["Authentication"])
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])

@app.on_event("startup")
async def startup_db_client():
    if not initialize_db():
        logger.error("Failed to initialize database connection")

@app.on_event("shutdown")
async def shutdown_db_client():
    close_db_connection()

# --- Pydantic Models ---

# --- Nested models for ParsedResumeData ---
class PassportDetailsModel(BaseModel):
    number: Optional[str] = ""
    expiryDate: Optional[str] = ""
    issuingCountry: Optional[str] = ""

class PersonalDetailsModel(BaseModel):
    name: Optional[str] = ""
    phone: Optional[str] = ""
    DOB: Optional[str] = ""
    gender: Optional[str] = ""
    maritalStatus: Optional[str] = ""
    nationality: Optional[str] = ""
    country_residency: Optional[str] = ""
    passportDetails: Optional[PassportDetailsModel] = Field(default_factory=PassportDetailsModel)

class QuestionAnswerModel(BaseModel):
    question: Optional[str] = ""
    answer: Optional[str] = ""

class AustralianEducationItemModel(BaseModel):
    degree: Optional[str] = ""
    fieldOfStudy: Optional[str] = ""
    institution: Optional[str] = ""
    country: Optional[str] = ""
    institutionPostCode: Optional[str] = ""
    commencementDate: Optional[str] = ""
    completionDate: Optional[str] = ""
    duration_in_years: Optional[str] = ""

class OverseasEducationItemModel(BaseModel):
    degree: Optional[str] = ""
    fieldOfStudy: Optional[str] = ""
    institution: Optional[str] = ""
    country: Optional[str] = ""
    commencementDate: Optional[str] = ""
    completionDate: Optional[str] = ""
    duration_in_years: Optional[str] = ""

class EducationDetailsModel(BaseModel):
    hasStudiedInAustralia: Optional[str] = ""
    numberOfQualificationsCompletedInAustralia: Optional[str] = ""
    questions_if_has_studied_in_australia: List[QuestionAnswerModel] = []
    australian_education: List[AustralianEducationItemModel] = []
    overseas_education: List[OverseasEducationItemModel] = []

class AustralianWorkExperienceItemModel(BaseModel):
    company: Optional[str] = ""
    role: Optional[str] = ""
    startDate: Optional[str] = ""
    endDate: Optional[str] = ""
    country: Optional[str] = ""
    postalCode: Optional[str] = ""

class OverseasWorkExperienceItemModel(BaseModel):
    company: Optional[str] = ""
    role: Optional[str] = ""
    startDate: Optional[str] = ""
    endDate: Optional[str] = ""
    country: Optional[str] = ""

class WorkExperienceDetailsModel(BaseModel):
    australian_workExperience: List[AustralianWorkExperienceItemModel] = []
    overseas_workExperience: List[OverseasWorkExperienceItemModel] = []

class EnglishExamDetailsModel(BaseModel):
    examName: Optional[str] = ""
    examDate: Optional[str] = ""
    overallScore: Optional[str] = ""
    listeningScore: Optional[str] = ""
    readingScore: Optional[str] = ""
    speakingScore: Optional[str] = ""
    writingScore: Optional[str] = ""

class EnglishProficiencyDetailsModel(BaseModel):
    englishLanguageTestCompleted: Optional[str] = ""
    englishExamDetails: Optional[EnglishExamDetailsModel] = Field(default_factory=EnglishExamDetailsModel)
    estimatedProficiency: Optional[str] = ""

class PartnerWorkExperienceModel(BaseModel):
    australian_workExperience: Optional[List[AustralianWorkExperienceItemModel]] = Field(default_factory=list)
    overseas_workExperience: Optional[List[OverseasWorkExperienceItemModel]] = Field(default_factory=list)

class IfPartnerModel(BaseModel):
    work_experience_in_last_5_years: Optional[PartnerWorkExperienceModel] = Field(
        default_factory=PartnerWorkExperienceModel, # This expects an object/dict
        alias="workexperiance_in_last_5_years",
        description="Partner's work experience in the last 5 years"
    )
    estimatedProficiency: Optional[str] = Field(None, description="Partner's estimated English proficiency")

class CommunityLanguageAccreditationModel(BaseModel):
    holdsNAATIcertification: Optional[str] = ""
    canGiveCommunityLanguageClasses: Optional[str] = ""

class StateLivedInModel(BaseModel):
    stateName: Optional[str] = Field(None)
    postCode: Optional[str] = Field(None)
    # 'from' is a Python keyword, so an alias is necessary.
    from_date: Optional[str] = Field(None, alias="from", serialization_alias="from", description="Start date of living in the state")
    to_date: Optional[str] = Field(None, alias="to", serialization_alias="to", description="End date of living in the state")

class LivingInAustraliaModel(BaseModel):
    hasLivedInAustralia: Optional[str] = Field(None, description="Has the candidate lived in Australia? (Yes/No)")
    # Schema key "numeberOfDiffrentStatesLivedIn" (typo)
    number_of_different_states_lived_in: Optional[str] = Field(
        None,
        alias="numeberOfDiffrentStatesLivedIn", # Matches schema key with typo
        description="Number of different states lived in Australia"
    )
    # Schema key "satesLivedIn" (typo)
    states_lived_in: Optional[List[StateLivedInModel]] = Field(
        default_factory=list,
        alias="satesLivedIn", # Matches schema key with typo
        description="Details of states lived in Australia"
    )

# Base resume data structure matching the Gemini prompt schema
class ParsedResumeDataBase(BaseModel):
    email: Optional[EmailStr] = Field(None, description="Candidate's email, crucial for identification and updates")
    personal_details: Optional[PersonalDetailsModel] = Field(default_factory=PersonalDetailsModel)
    education: Optional[EducationDetailsModel] = Field(default_factory=EducationDetailsModel)
    workExperience: Optional[WorkExperienceDetailsModel] = Field(default_factory=WorkExperienceDetailsModel)
    englishProficiency: Optional[EnglishProficiencyDetailsModel] = Field(default_factory=EnglishProficiencyDetailsModel)
    if_partner: Optional[IfPartnerModel] = Field(default_factory=IfPartnerModel, description="Details if the candidate has a partner")
    community_language_accreditation: Optional[CommunityLanguageAccreditationModel] = Field(default_factory=CommunityLanguageAccreditationModel)
    living_in_australia: Optional[LivingInAustraliaModel] = Field(default_factory=LivingInAustraliaModel, description="Details about living in Australia")

# Model for data stored in resume_collection
class ParsedResumeDataStorage(ParsedResumeDataBase):
    pass

# --- Occupation Suggestion Structure ---
class MatchedOccupationDetail(BaseModel):
    matchedOccupation: str
    matchedANZSCO: str
    similarityScore: float
    basedOnSuggestion: str

class OccupationSuggestionStorage(BaseModel):
    uploader_email: str
    candidate_email: EmailStr
    suggested_by_llm: List[str] = Field(default=[], description="Occupations suggested by the LLM")
    matched_details: List[MatchedOccupationDetail] = Field(default=[], description="ANZSCO matches based on embedding similarity")
    last_updated: Optional[str] = None

# --- API Response Models ---
class ApiResponse(BaseModel):
    uploader_email: str
    candidate_email: Optional[EmailStr] = None
    parsed_data_summary: Optional[Dict[str, Any]] = Field(None, description="Basic info like name, email extracted")
    status: str
    resume_storage_status: str
    occupation_suggestion_status: str
    message: Optional[str] = None

class CandidateDataResponse(BaseModel):
    uploader_email: str
    candidate_email: EmailStr
    candidate_data: ParsedResumeDataStorage

class OccupationSuggestionResponse(BaseModel):
    uploader_email: str
    candidate_email: EmailStr
    suggestions: OccupationSuggestionStorage

    @classmethod
    def from_storage(cls, storage: OccupationSuggestionStorage):
        # Deduplicate matched_details based on anzsco_code
        seen_codes = set()
        unique_matched_details = []
        for detail in storage.matched_details:
            if detail.anzsco_code not in seen_codes:
                seen_codes.add(detail.anzsco_code)
                unique_matched_details.append(detail)

        # Create a new OccupationSuggestionStorage with unique matched_details
        unique_suggestions = OccupationSuggestionStorage(
            uploader_email=storage.uploader_email,
            candidate_email=storage.candidate_email,
            suggested_by_llm=storage.suggested_by_llm,  # Keep LLM suggestions as is
            matched_details=unique_matched_details,
            last_updated=storage.last_updated
        )

        return cls(
            uploader_email=storage.uploader_email,
            candidate_email=storage.candidate_email,
            suggestions=unique_suggestions
        )

    @classmethod
    def from_storage(cls, storage: OccupationSuggestionStorage):
        # Deduplicate matched_details based on anzsco_code
        seen_codes = set()
        unique_matched_details = []
        for detail in storage.matched_details:
            if detail.anzsco_code not in seen_codes:
                seen_codes.add(detail.anzsco_code)
                unique_matched_details.append(detail)

        # Create a new OccupationSuggestionStorage with unique matched_details
        unique_suggestions = OccupationSuggestionStorage(
            uploader_email=storage.uploader_email,
            candidate_email=storage.candidate_email,
            suggested_by_llm=storage.suggested_by_llm,  # Keep LLM suggestions as is
            matched_details=unique_matched_details,
            last_updated=storage.last_updated
        )

        return cls(
            uploader_email=storage.uploader_email,
            candidate_email=storage.candidate_email,
            suggestions=unique_suggestions
        )

# --- Models for PUT/Update ---
# UpdateCandidateDataRequest now directly uses the full ParsedResumeDataBase structure for replacement
class UpdateCandidateDataRequest(ParsedResumeDataBase):
    class Config:
        anystr_strip_whitespace = True

# --- Helper Functions ---
def extract_text_from_pdf(file_path: str, filename: str = "file") -> str:
    text = ""
    try:
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            if pdf_reader.is_encrypted:
                try:
                    if pdf_reader.decrypt('') == PyPDF2.PasswordType.WRONG_PASSWORD:
                        logger.warning(f"Encrypted PDF {filename} couldn't be decrypted.")
                except Exception as decrypt_err:
                    logger.warning(f"Error decrypting PDF {filename}: {decrypt_err}")

            num_pages = len(pdf_reader.pages)
            if num_pages == 0:
                logger.warning(f"PDF has 0 pages: {filename}")

            for i, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as page_err:
                    logger.error(f"Error extracting text from page {i+1} of {filename}: {page_err}")

            if not text.strip():
                 if num_pages > 0:
                     logger.warning(f"No text extracted from PDF: {filename}. May be image-based or empty.")
            return text
    except FileNotFoundError:
        logger.error(f"PDF file not found: {file_path}")
        raise Exception("PDF file not found.")
    except PyPDF2.errors.PdfReadError as pdf_err:
        logger.error(f"Invalid/corrupted PDF: {filename} - {pdf_err}")
        raise Exception(f"Invalid or corrupted PDF file: {filename}.")
    except ValueError as ve:
        raise ve
    except Exception as e:
        logger.exception(f"Error processing PDF {filename}: {e}")
        raise Exception(f"Error processing PDF file: {filename}.")

def parse_resume_with_gemini(text: str) -> tuple[Optional[Dict[str, Any]], Optional[List[str]]]:
    if not text or text.isspace():
        return None, None

    prompt = f"""
    Analyze the following resume text and perform the following tasks:

    ---

    #### 🎯 Your Goals:
    1. **Extract detailed information into a strictly structured JSON object** based on the schema provided.
    2. **Suggest 4–5 relevant occupation titles aligned with the Australian ANZSCO classification**, giving **higher weight to roles where the candidate has:**
    - **Longer durations of professional experience**
    - **Higher or more relevant educational qualifications**
    - **Work closely related to ANZSCO-recognized job roles**

    ---

    #### 📤 Output Requirements:
    - Return a **single, strictly valid JSON object** with **only** the following two top-level keys:
    - `"structured_data"`: Follows the schema below. Use `""` or `[]` if data is missing.
    - `"suggested_occupations"`: A list of 4–5 **broad occupation titles** (e.g., `"Software Engineer"`, `"Marketing Specialist"`, `"Registered Nurse"`) that are **plausibly mapped to ANZSCO roles**.
    - **Do not include any text, markdown, or comments before or after the JSON output.**
    - Ensure the `"email"` field captures the candidate’s **primary email address**, and all date fields are in `"DD/MM/YYYY"` format where available.
    - Include only higher education qualifications, Do not extract high school, secondary school

    ---

    #### 🧠 Occupation Suggestion Criteria:
    - Focus on ANZSCO-aligned broad titles.
    - Prefer occupations:
    - Where experience spans multiple years.
    - Where education aligns with occupational domains.
    - Where Australia-based experience exists (if applicable).
    - Avoid overly niche job titles; keep them general and relevant to ANZSCO.

    ---

    **Target JSON Schema for "structured_data":**
    {{
        "email": "string",
        "personal_details": {{
            "name": "string",
            "phone": "string",
            "DOB": "string", //DD/MM/YYYY
            "gender": "string",
            "maritalStatus": "string",
            "nationality": "string",
            "country_residency": "string",
            "passportDetails": {{
                "number": "string",
                "expiryDate": "string",
                "issuingCountry": "string"
            }}
        }},
        "education": {{
            "hasStudiedInAustralia": "string",
            "numberOfQualificationsCompletedInAustralia": "string",
            "questions_if_has_studied_in_australia": [
                {{
                    "question": "Did you complete at least 2 years of full-time study in Australia? (Yes/No)",
                    "answer": ""
                }},
                {{
                    "question": "Was this study completed in regional Australia (ask postcode)",
                    "answer": ""
                }},
                {{
                    "question": "Did you complete a Master’s by Research or PhD in a STEM field (Science,Technology, Engineering, Mathematics) from an Australian institution?",
                    "answer": ""
                }},
                {{
                    "question": "Have you completed a Professional Year program in Australia in the last 4 years? (Yes/No)",
                    "answer": ""
                }}
            ],
            "australian_education": [
                {{
                    "degree": "string", //Include only higher education qualifications, Do not extract high school, secondary school
                    "fieldOfStudy": "string",
                    "institution": "string",
                    "country": "string",
                    "institutionPostCode": "string",
                    "commencementDate": "string", // DD/MM/YYYY or MM/YYYY
                    "completionDate": "string", // DD/MM/YYYY or MM/YYYY
                    "duration_in_years": "string"
                }}
            ],
            "overseas_education": [
                {{
                    "degree": "string",  //Include only higher education qualifications, Do not extract high school, secondary school
                    "fieldOfStudy": "string",
                    "institution": "string",
                    "country": "string",
                    "commencementDate": "string", // DD/MM/YYYY or MM/YYYY
                    "completionDate": "string", // DD/MM/YYYY or MM/YYYY
                    "duration_in_years": "string"
                }}
            ]
        }},
        "workExperience": {{
            "australian_workExperience": [
                {{
                    "company": "string",
                    "role": "string",
                    "startDate": "string", // DD/MM/YYYY or MM/YYYY
                    "endDate": "string", // DD/MM/YYYY or MM/YYYY
                    "country": "string",
                    "postalCode": "string"
                }}
            ],
            "overseas_workExperience": [
                {{
                    "company": "string",
                    "role": "string",
                    "startDate": "string", // DD/MM/YYYY or MM/YYYY
                    "endDate": "string", // DD/MM/YYYY or MM/YYYY
                    "country": "string"
                }}
            ]
        }},
        "englishProficiency": {{
            "englishLanguageTestCompleted": "string",
            "englishExamDetails": {{
                "examName": "string",
                "examDate": "string", // DD/MM/YYYY or MM/YYYY
                "expiryDate": "string",  // DD/MM/YYYY or MM/YYYY
                "overallScore": "string",
                "listeningScore": "string",
                "readingScore": "string",
                "speakingScore": "string",
                "writingScore": "string"
            }},
            "estimatedProficiency": "string"
        }},
        "if_partner": {{
            "name": "string",
            "occupation": "string",
            "isAgeBelow45": "string",
            "hasCompetentEnglish": "string",
            "workexperiance_in_last_5_years": {{
                "australian_workExperience": [
                    {{
                        "company": "string",
                        "role": "string",
                        "startDate": "string", //DD/MM/YYYY or MM/YYYY
                        "endDate": "string", //DD/MM/YYYY or MM/YYYY
                        "country": "string",
                        "postalCode": "string"
                    }}
                ],
                "overseas_workExperience": [
                    {{
                        "company": "string",
                        "role": "string",
                        "startDate": "string", //DD/MM/YYYY or MM/YYYY
                        "endDate": "string", //DD/MM/YYYY or MM/YYYY
                        "country": "string"
                    }}
                ]
            }}
        }},
        "community_language_accreditation": {{
            "holdsNAATIcertification": "string",
            "canGiveCommunityLanguageClasses": "string"
        }},
        "living_in_australia": {{
            "currentSuburb": "string",
            "currentPostCode": "string",
            "currentState": "string",
            "currentVisaStatus": "string",
            "hasLivedInAustralia": "string",
            "numberOfDifferentStatesLivedIn": "string",
            "satesLivedIn": [
                {{
                    "stateName": "string",
                    "postCode": "string",
                    "from": "string",
                    "to": "string"
                }}
            ]
        }}
    }}

        **Complete Output Example Structure:**
        {{
        "structured_data": {{
            "email": "jane.doe@example.com",
            "personal_details": {{
                "name": "Jane Doe"
            }}
            // ... other structured fields ...
        }},
        "suggested_occupations": [
            "Occupation Title 1",
            "Occupation Title 2",
            "Occupation Title 3",
            "Occupation Title 4"
        ]
        }}

    **Resume Text to Parse:**
    --- START ---
    {text}
    --- END ---

    **Output (Valid JSON Object ONLY):**
    """

    try:
        response = gemini_client.models.generate_content(
            model=f"models/{gemini_model_name}",
            contents=prompt,
        )
        content = response.text

        if not content:
             raise Exception(f"Gemini API ({gemini_model_name}) returned no content.")

        content_cleaned = content.strip()
        if content_cleaned.startswith("```json"):
            content_cleaned = content_cleaned[7:].rstrip("` \n")
        elif content_cleaned.startswith("```"):
             content_cleaned = content_cleaned[3:].rstrip("` \n")
        content_cleaned = content_cleaned.strip()

        if not content_cleaned.startswith("{") or not content_cleaned.endswith("}"):
             logger.warning(f"Gemini response may not be valid JSON structure:\n{content_cleaned[:100]}...")

        try:
            parsed_response = json.loads(content_cleaned)
            if not isinstance(parsed_response, dict):
                raise ValueError("Parsed response is not a dictionary.")

            structured_data = parsed_response.get("structured_data")
            suggested_occupations = parsed_response.get("suggested_occupations")

            if not isinstance(structured_data, dict):
                logger.error("Gemini response missing or invalid 'structured_data' field.")
                raise ValueError("Missing or invalid 'structured_data' in Gemini response.")

            candidate_email = structured_data.get("email")
            if not candidate_email or not isinstance(candidate_email, str) or "@" not in candidate_email:
                 logger.error("Mandatory 'email' field missing or invalid in 'structured_data'.")
                 raise ValueError("Mandatory 'email' field missing or invalid in parsed data.")

            if suggested_occupations is None:
                logger.warning("Gemini response missing 'suggested_occupations' field. Proceeding without suggestions.")
                suggested_occupations = []
            elif not isinstance(suggested_occupations, list):
                logger.warning("Gemini response 'suggested_occupations' is not a list. Ignoring suggestions.")
                suggested_occupations = []

            logger.info("Successfully parsed resume text and suggestions using Gemini.")
            return structured_data, suggested_occupations

        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to decode JSON from Gemini. Error: {json_err}. Content:\n{content_cleaned}")
            raise Exception(f"Gemini response was not valid JSON: {json_err}")
        except ValueError as ve:
            logger.error(f"Validation Error in Gemini response structure: {ve}")
            raise ve

    except Exception as e:
        logger.exception(f"Error during Gemini API call ({gemini_model_name}) or processing: {e}")
        raise Exception(f"Error parsing resume with Gemini: {e}")


def get_gemini_embedding(text: str, api_key: str) -> Optional[List[float]]:
    if not text or text.isspace():
        logger.warning("Attempted to get embedding for empty text.")
        return None

    url = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    data = {
        "model": "models/text-embedding-004",
        "content": {"parts": [{"text": text}]}
    }

    try:
        response = requests.post(url, headers=headers, params=params, json=data, timeout=20)
        response.raise_for_status()

        result = response.json()
        embedding = result.get("embedding", {}).get("values")
        if embedding and isinstance(embedding, list):
             return embedding
        else:
             logger.error(f"Failed to extract valid embedding from API response for text: '{text[:50]}...'. Response: {result}")
             return None
    except requests.exceptions.RequestException as req_err:
        logger.error(f"API request failed for embedding: {req_err}. URL: {url}")
        logger.error(f"Response status: {req_err.response.status_code if req_err.response else 'N/A'}, Text: {req_err.response.text if req_err.response else 'N/A'}")
        return None
    except Exception as e:
        logger.exception(f"Error getting Gemini embedding for text '{text[:50]}...': {e}")
        return None

def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    if not embedding1 or not embedding2:
        return 0.0
    a = np.array(embedding1)
    b = np.array(embedding2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def find_best_occupation_matches(
    suggested_occupations: List[str],
    loaded_occupations: List[Dict[str, Any]],
    api_key: str,
    similarity_threshold: float = SIMILARITY_THRESHOLD
    ) -> List[MatchedOccupationDetail]:
    if not suggested_occupations or not loaded_occupations:
        logger.info("Skipping occupation matching: No suggestions or no loaded occupation data.")
        return []

    if not api_key:
         logger.error("Cannot perform occupation matching: GOOGLE_API_KEY is missing.")
         return []

    matched_results = []
    processed_suggestions = set()

    for suggestion in suggested_occupations:
        if not suggestion or not isinstance(suggestion, str) or not suggestion.strip() or suggestion in processed_suggestions:
            continue
        processed_suggestions.add(suggestion)

        suggestion_embedding = get_gemini_embedding(suggestion, api_key)
        if not suggestion_embedding:
            logger.warning(f"Couldn't get embedding for suggestion: '{suggestion}'")
            continue

        best_match_for_suggestion = None
        highest_similarity_for_suggestion = -1

        for occupation in loaded_occupations:
            try:
                similarity = cosine_similarity(suggestion_embedding, occupation["embedding"])
                if similarity > highest_similarity_for_suggestion:
                    highest_similarity_for_suggestion = similarity
                    best_match_for_suggestion = occupation
            except Exception as e:
                logger.error(f"Error computing similarity for '{occupation.get('Occupation', 'Unknown')}' vs '{suggestion}': {e}")
                continue

        if best_match_for_suggestion and highest_similarity_for_suggestion >= similarity_threshold:
            logger.info(f"Match found: '{best_match_for_suggestion['Occupation']}' (ANZSCO: {best_match_for_suggestion['ANZSCO']}) for suggestion '{suggestion}' with similarity {highest_similarity_for_suggestion:.4f}")
            matched_results.append(MatchedOccupationDetail(
                matchedOccupation=best_match_for_suggestion["Occupation"],
                matchedANZSCO=best_match_for_suggestion["ANZSCO"],
                similarityScore=highest_similarity_for_suggestion,
                basedOnSuggestion=suggestion
            ))
        else:
            logger.info(f"No suitable match found (Score < {similarity_threshold}) for suggestion '{suggestion}'. Best similarity: {highest_similarity_for_suggestion:.4f}")
    return matched_results


def store_resume_data(parsed_data: Dict[str, Any], uploader_email: str) -> tuple[str, Optional[str], Optional[str]]:
    uploader_email_lower = uploader_email.lower()
    candidate_email = parsed_data.get("email")

    if not candidate_email:
        logger.error(f"Critical Error: Candidate email missing in parsed data before storage. Uploader: {uploader_email_lower}")
        return "failed", "Internal Error: Candidate email missing for storage.", None

    candidate_email_lower = candidate_email.lower()
    logger.info(f"Storing/updating candidate resume '{candidate_email_lower}' for uploader '{uploader_email_lower}' in collection '{RESUME_COLLECTION_NAME}'")

    try:
        validated_storage_data = ParsedResumeDataStorage(**parsed_data).dict(exclude_none=True) # Use exclude_none to match Pydantic defaults
    except Exception as pyd_err:
         logger.error(f"Pydantic validation failed before storing resume data for {candidate_email_lower}: {pyd_err}")
         return "failed", f"Data validation failed before storage: {pyd_err}", candidate_email_lower

    candidate_entry = {
        "candidate_email": candidate_email_lower,
        "parsed_resume_data": validated_storage_data
    }

    try:
        update_result = resume_collection.update_one(
            {"_id": uploader_email_lower, "candidates.candidate_email": candidate_email_lower},
            {"$set": {"candidates.$": candidate_entry}}
        )

        if update_result.matched_count > 0:
            status = "updated" if update_result.modified_count > 0 else "no_change"
            message = f"Resume {status} for candidate: {candidate_email}."
            logger.info(f"{status.capitalize()} candidate resume '{candidate_email_lower}' for uploader '{uploader_email_lower}'.")
            return status, message, candidate_email_lower
        else:
            add_result = resume_collection.update_one(
                {"_id": uploader_email_lower},
                {
                    "$push": {"candidates": candidate_entry},
                    "$setOnInsert": {"uploader_email": uploader_email_lower}
                },
                upsert=True
            )
            status = "created"
            if add_result.upserted_id:
                 message = f"New uploader record created. Added resume for candidate: {candidate_email}."
                 logger.info(f"Created uploader '{uploader_email_lower}' and added candidate resume '{candidate_email_lower}'.")
            elif add_result.modified_count > 0:
                 message = f"Added new resume for candidate: {candidate_email}."
                 logger.info(f"Added new candidate resume '{candidate_email_lower}' to uploader '{uploader_email_lower}'.")
            else:
                 status = "failed"
                 message = "Database reported no change trying to add candidate resume."
                 logger.warning(f"DB reported no modification attempting to add candidate resume '{candidate_email_lower}' for uploader '{uploader_email_lower}'. Result: {add_result.raw_result}")
            return status, message, candidate_email_lower

    except OperationFailure as op_err:
        logger.error(f"MongoDB operation failed storing resume for uploader '{uploader_email_lower}', candidate '{candidate_email_lower}': {op_err.details}", exc_info=True)
        return "failed", f"Database operation error: {op_err.code_name}", candidate_email_lower
    except Exception as e:
        logger.exception(f"Error storing resume data for uploader '{uploader_email_lower}', candidate '{candidate_email_lower}': {e}")
        return "failed", "An unexpected error occurred during resume data storage.", candidate_email_lower


def store_occupation_suggestions(
    uploader_email: str,
    candidate_email: str,
    suggested_by_llm: List[str],
    matched_details: List[MatchedOccupationDetail]
    ) -> tuple[str, str]:
    uploader_email_lower = uploader_email.lower()
    candidate_email_lower = candidate_email.lower()
    storage_id = f"{uploader_email_lower}_{candidate_email_lower}"

    logger.info(f"Storing/updating occupation suggestions for candidate '{candidate_email_lower}' (Uploader: '{uploader_email_lower}') in collection '{OCCUPATION_COLLECTION_NAME}'")

    try:
        suggestion_doc = OccupationSuggestionStorage(
            uploader_email=uploader_email_lower,
            candidate_email=candidate_email_lower,
            suggested_by_llm=suggested_by_llm,
            matched_details=matched_details,
            last_updated=datetime.utcnow().isoformat() + "Z"
        )
        doc_to_store = suggestion_doc.dict(exclude_none=True)

        update_result = occupation_collection.update_one(
            {"_id": storage_id},
            {"$set": doc_to_store},
            upsert=True
        )

        if update_result.upserted_id:
            status = "created"
            message = "Occupation suggestions stored."
            logger.info(f"Stored new occupation suggestions for ID: {storage_id}")
        elif update_result.matched_count > 0:
            status = "updated" if update_result.modified_count > 0 else "no_change"
            message = f"Occupation suggestions {status}."
            logger.info(f"Occupation suggestions {status} for ID: {storage_id}")
        else:
             status = "failed"
             message = "Failed to store occupation suggestions (unknown reason)."
             logger.error(f"Unexpected result storing occupation suggestions for {storage_id}. Result: {update_result.raw_result}")
        return status, message

    except OperationFailure as op_err:
        logger.error(f"MongoDB operation failed storing occupation suggestions for {storage_id}: {op_err.details}", exc_info=True)
        return "failed", f"Database operation error storing suggestions: {op_err.code_name}"
    except Exception as e:
        logger.exception(f"Error storing occupation suggestions for {storage_id}: {e}")
        return "failed", "An unexpected error occurred during suggestion storage."

# --- API Endpoints ---

def extract_text_from_docx(file_path: str, filename: str) -> str:
    try:
        logger.info(f"Extracting text from DOCX: {filename}")
        with open(file_path, "rb") as docx_file:
            result = mammoth.extract_raw_text(docx_file)
            text = result.value
            logger.info(f"Successfully extracted {len(text)} characters from {filename}.")
            return text
    except Exception as e:
        logger.error(f"Failed to extract text from DOCX file {filename}: {e}")
        # Re-raise as a ValueError to be caught by the main exception handler
        raise ValueError(f"Could not process the Word document '{filename}'. It might be corrupted or in an unsupported format.")

# @app.post("/api/parse/file", response_model=ApiResponse, status_code=200, tags=["Parsing & Storage"])
# async def parse_file_and_suggest(file: UploadFile = File(...), uploader_email: str = Form(...),
#     current_user: Dict[str, Any] = Depends(get_current_user)
# ):
#     # Verify if the current user has access to this data
#     if not verify_access(current_user, uploader_email):
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="You don't have permission to access this data"
#         )
#     temp_file_path = None
#     structured_data: Optional[Dict[str, Any]] = None
#     suggested_occupations: Optional[List[str]] = []
#     matched_occupation_details: List[MatchedOccupationDetail] = []
#     resume_storage_status = "not_attempted"
#     occupation_suggestion_status = "not_attempted"
#     db_message_resume = "Resume storage not attempted."
#     db_message_occupation = "Occupation suggestion storage not attempted."
#     candidate_email_processed: Optional[str] = None
#     status = "error"
#     final_message = "Processing failed."
#     filename = file.filename or "unknown.pdf"

#     if not filename.lower().endswith('.pdf'):
#         raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are supported.")
#     if not uploader_email or "@" not in uploader_email:
#          raise HTTPException(status_code=400, detail="Valid uploader_email is required.")

#     try:
#         logger.info(f"Request from '{uploader_email}' to parse file: {filename}")

#         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as disk_temp_file:
#             temp_file_path = disk_temp_file.name
#             content = await file.read()
#             if not content:
#                  raise HTTPException(status_code=400, detail="Uploaded file is empty.")
#             if len(content) > MAX_FILE_SIZE:
#                  raise HTTPException(status_code=413, detail=f"File size exceeds {MAX_FILE_SIZE / (1024*1024):.1f} MB limit.")
#             disk_temp_file.write(content)
#             logger.info(f"Saved {filename} temporarily to: {temp_file_path}")

#         logger.info(f"Extracting text from PDF: {filename}")
#         text = extract_text_from_pdf(temp_file_path, filename)
#         if not text.strip():
#             logger.warning(f"No text could be extracted from {filename}. Aborting parsing.")
#             raise HTTPException(status_code=400, detail="Could not extract text from the PDF (possibly image-based or empty).")

#         logger.info(f"Parsing extracted text with Gemini for: {filename}")
#         structured_data, suggested_occupations = parse_resume_with_gemini(text)

#         if not structured_data or not structured_data.get("email"):
#             status = "parsing_failed"
#             final_message = "Failed to parse resume or extract mandatory candidate email."
#             logger.error(f"Parsing failed or email missing for {filename}")
#             # Validate with Pydantic before raising HTTPException to get more specific error
#             try:
#                 ParsedResumeDataStorage(**(structured_data or {}))
#             except Exception as pyd_err:
#                  final_message = f"Failed to parse resume or structure is invalid: {pyd_err}"
#                  logger.error(f"Pydantic validation failed for parsed data from {filename}: {pyd_err}")
#             raise HTTPException(status_code=500, detail=final_message)

#         candidate_email_processed = structured_data.get("email")
#         status = "success"

#         logger.info(f"Storing parsed resume data in MongoDB for: {filename}")
#         resume_storage_status, db_message_resume, _ = store_resume_data(structured_data, uploader_email)
#         if resume_storage_status == "failed":
#              logger.error(f"Failed to store resume data for {candidate_email_processed}. Message: {db_message_resume}")
#              status = "partial_success"
#              final_message = f"Parsing succeeded, but failed to store resume data. {db_message_resume}"

#         if suggested_occupations and occupation_data:
#             logger.info(f"Performing occupation matching for {len(suggested_occupations)} suggestions from {filename}")
#             occupation_suggestion_status = "processing"
#             matched_occupation_details = find_best_occupation_matches(
#                 suggested_occupations,
#                 occupation_data,
#                 GOOGLE_API_KEY
#             )
#             occupation_suggestion_status = "matching_complete"
#             if not matched_occupation_details:
#                  logger.info(f"No ANZSCO matches found meeting threshold for suggestions from {filename}")
#         elif not occupation_data:
#             logger.warning("Occupation matching skipped: Embeddings data not loaded.")
#             occupation_suggestion_status = "disabled"
#         else:
#             logger.info("No occupation suggestions provided by LLM.")
#             occupation_suggestion_status = "no_suggestions"


#         if suggested_occupations or matched_occupation_details:
#             logger.info(f"Storing occupation suggestions/matches in MongoDB for: {filename}")
#             occ_store_status, db_message_occupation = store_occupation_suggestions(
#                 uploader_email,
#                 candidate_email_processed,
#                 suggested_occupations or [],
#                 matched_occupation_details
#             )
#             occupation_suggestion_status = occ_store_status
#             if occ_store_status == "failed":
#                 logger.error(f"Failed to store occupation suggestions for {candidate_email_processed}. Message: {db_message_occupation}")
#                 if status == "success": status = "partial_success"
#                 final_message += f" Failed to store occupation suggestions. {db_message_occupation}"
#         else:
#              db_message_occupation = "No suggestions or matches to store."

#         if status == "success":
#              final_message = "Resume processed and suggestions generated successfully."
#         elif status == "partial_success" and not final_message.startswith("Parsing succeeded"): # If not already set by resume store failure
#              final_message = f"Processing partially successful. Check details. Resume Store: {db_message_resume} Suggestion Store: {db_message_occupation}"
        
#         parsed_data_summary_name = ""
#         if structured_data and structured_data.get("personal_details"):
#             parsed_data_summary_name = structured_data.get("personal_details", {}).get("name")


#         parsed_data_summary = {
#              "name": parsed_data_summary_name,
#              "email": candidate_email_processed,
#              "suggestions_count": len(suggested_occupations or []),
#              "matches_found": len(matched_occupation_details)
#          }

#         return ApiResponse(
#             uploader_email=uploader_email,
#             candidate_email=candidate_email_processed,
#             parsed_data_summary=parsed_data_summary,
#             status=status,
#             resume_storage_status=resume_storage_status,
#             occupation_suggestion_status=occupation_suggestion_status,
#             message=final_message
#         )

#     except HTTPException as e:
#         logger.error(f"HTTP Error for {uploader_email}, file {filename}: {e.detail}")
#         return JSONResponse(
#             status_code=e.status_code,
#             content=ApiResponse(
#                 uploader_email=uploader_email,
#                 candidate_email=candidate_email_processed,
#                 status="error",
#                 resume_storage_status=resume_storage_status,
#                 occupation_suggestion_status=occupation_suggestion_status,
#                 message=e.detail
#             ).dict(exclude_none=True)
#         )
#     except (ValueError, json.JSONDecodeError, PyPDF2.errors.PdfReadError) as ve:
#          logger.error(f"Processing Error for {uploader_email}, file {filename}: {ve}", exc_info=False)
#          error_code = 422
#          err_status_detail = "parsing_failed" if "Gemini" in str(ve) or "JSON" in str(ve) else "extraction_failed"
#          if isinstance(ve, PyPDF2.errors.PdfReadError): error_code = 400

#          return JSONResponse(
#              status_code=error_code,
#              content=ApiResponse(
#                  uploader_email=uploader_email,
#                  candidate_email=candidate_email_processed,
#                  status=err_status_detail,
#                  resume_storage_status=resume_storage_status,
#                  occupation_suggestion_status=occupation_suggestion_status,
#                  message=f"Failed to process resume: {ve}"
#              ).dict(exclude_none=True)
#          )
#     except Exception as e:
#         logger.exception(f"Unexpected Internal Error for {uploader_email}, file {filename}: {e}")
#         return JSONResponse(
#              status_code=500,
#              content=ApiResponse(
#                  uploader_email=uploader_email,
#                  candidate_email=candidate_email_processed,
#                  status="internal_error",
#                  resume_storage_status=resume_storage_status,
#                  occupation_suggestion_status=occupation_suggestion_status,
#                  message="An unexpected internal server error occurred."
#              ).dict(exclude_none=True)
#         )
#     finally:
#         if temp_file_path and os.path.exists(temp_file_path):
#             try:
#                 os.unlink(temp_file_path)
#             except Exception as unlink_err:
#                  logger.error(f"Error deleting temp file {temp_file_path}: {unlink_err}")
#         if file:
#              await file.close()

@app.post("/api/parse/file", response_model=ApiResponse, status_code=200, tags=["Parsing & Storage"])
async def parse_file_and_suggest(file: UploadFile = File(...), uploader_email: str = Form(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    # Verify if the current user has access to this data
    if not verify_access(current_user, uploader_email):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this data"
        )
        
    temp_file_path = None
    structured_data: Optional[Dict[str, Any]] = None
    suggested_occupations: Optional[List[str]] = []
    matched_occupation_details: List[MatchedOccupationDetail] = []
    resume_storage_status = "not_attempted"
    occupation_suggestion_status = "not_attempted"
    db_message_resume = "Resume storage not attempted."
    db_message_occupation = "Occupation suggestion storage not attempted."
    candidate_email_processed: Optional[str] = None
    status = "error"
    final_message = "Processing failed."
    filename = file.filename or "unknown.file" # CHANGED: More generic default

    # NEW: Define allowed extensions and validate the uploaded file
    allowed_extensions = {".pdf", ".docx"}
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF (.pdf) and Word (.docx) files are supported.")
        
    if not uploader_email or "@" not in uploader_email:
         raise HTTPException(status_code=400, detail="Valid uploader_email is required.")

    try:
        logger.info(f"Request from '{uploader_email}' to parse file: {filename}")

        # CHANGED: Use the dynamic file extension for the temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as disk_temp_file:
            temp_file_path = disk_temp_file.name
            content = await file.read()
            if not content:
                 raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            if len(content) > MAX_FILE_SIZE:
                 raise HTTPException(status_code=413, detail=f"File size exceeds {MAX_FILE_SIZE / (1024*1024):.1f} MB limit.")
            disk_temp_file.write(content)
            logger.info(f"Saved {filename} temporarily to: {temp_file_path}")

        # NEW: Conditional text extraction based on file type
        text = ""
        if file_ext == '.pdf':
            logger.info(f"Extracting text from PDF: {filename}")
            text = extract_text_from_pdf(temp_file_path, filename)
        elif file_ext == '.docx':
            logger.info(f"Extracting text from DOCX: {filename}")
            text = extract_text_from_docx(temp_file_path, filename)

        if not text.strip():
            logger.warning(f"No text could be extracted from {filename}. Aborting parsing.")
            raise HTTPException(status_code=400, detail=f"Could not extract text from the {file_ext.upper()} file (possibly image-based or empty).")

        # --- The rest of your logic remains exactly the same ---
        # It operates on the 'text' variable, which is now populated from either PDF or DOCX.

        logger.info(f"Parsing extracted text with Gemini for: {filename}")
        structured_data, suggested_occupations = parse_resume_with_gemini(text)

        if not structured_data or not structured_data.get("email"):
            status = "parsing_failed"
            final_message = "Failed to parse resume or extract mandatory candidate email."
            logger.error(f"Parsing failed or email missing for {filename}")
            try:
                ParsedResumeDataStorage(**(structured_data or {}))
            except Exception as pyd_err:
                 final_message = f"Failed to parse resume or structure is invalid: {pyd_err}"
                 logger.error(f"Pydantic validation failed for parsed data from {filename}: {pyd_err}")
            raise HTTPException(status_code=500, detail=final_message)

        candidate_email_processed = structured_data.get("email")
        status = "success"

        logger.info(f"Storing parsed resume data in MongoDB for: {filename}")
        resume_storage_status, db_message_resume, _ = store_resume_data(structured_data, uploader_email)
        if resume_storage_status == "failed":
             logger.error(f"Failed to store resume data for {candidate_email_processed}. Message: {db_message_resume}")
             status = "partial_success"
             final_message = f"Parsing succeeded, but failed to store resume data. {db_message_resume}"

        if suggested_occupations and occupation_data:
            logger.info(f"Performing occupation matching for {len(suggested_occupations)} suggestions from {filename}")
            occupation_suggestion_status = "processing"
            matched_occupation_details = find_best_occupation_matches(
                suggested_occupations,
                occupation_data,
                GOOGLE_API_KEY
            )
            occupation_suggestion_status = "matching_complete"
            if not matched_occupation_details:
                 logger.info(f"No ANZSCO matches found meeting threshold for suggestions from {filename}")
        elif not occupation_data:
            logger.warning("Occupation matching skipped: Embeddings data not loaded.")
            occupation_suggestion_status = "disabled"
        else:
            logger.info("No occupation suggestions provided by LLM.")
            occupation_suggestion_status = "no_suggestions"


        if suggested_occupations or matched_occupation_details:
            logger.info(f"Storing occupation suggestions/matches in MongoDB for: {filename}")
            occ_store_status, db_message_occupation = store_occupation_suggestions(
                uploader_email,
                candidate_email_processed,
                suggested_occupations or [],
                matched_occupation_details
            )
            occupation_suggestion_status = occ_store_status
            if occ_store_status == "failed":
                logger.error(f"Failed to store occupation suggestions for {candidate_email_processed}. Message: {db_message_occupation}")
                if status == "success": status = "partial_success"
                final_message += f" Failed to store occupation suggestions. {db_message_occupation}"
        else:
             db_message_occupation = "No suggestions or matches to store."

        if status == "success":
             final_message = "Resume processed and suggestions generated successfully."
        elif status == "partial_success" and not final_message.startswith("Parsing succeeded"):
             final_message = f"Processing partially successful. Check details. Resume Store: {db_message_resume} Suggestion Store: {db_message_occupation}"
        
        parsed_data_summary_name = ""
        if structured_data and structured_data.get("personal_details"):
            parsed_data_summary_name = structured_data.get("personal_details", {}).get("name")


        parsed_data_summary = {
             "name": parsed_data_summary_name,
             "email": candidate_email_processed,
             "suggestions_count": len(suggested_occupations or []),
             "matches_found": len(matched_occupation_details)
         }

        return ApiResponse(
            uploader_email=uploader_email,
            candidate_email=candidate_email_processed,
            parsed_data_summary=parsed_data_summary,
            status=status,
            resume_storage_status=resume_storage_status,
            occupation_suggestion_status=occupation_suggestion_status,
            message=final_message
        )

    except HTTPException as e:
        logger.error(f"HTTP Error for {uploader_email}, file {filename}: {e.detail}")
        return JSONResponse(
            status_code=e.status_code,
            content=ApiResponse(
                uploader_email=uploader_email,
                candidate_email=candidate_email_processed,
                status="error",
                resume_storage_status=resume_storage_status,
                occupation_suggestion_status=occupation_suggestion_status,
                message=e.detail
            ).dict(exclude_none=True)
        )
    except (ValueError, json.JSONDecodeError, PyPDF2.errors.PdfReadError) as ve:
         logger.error(f"Processing Error for {uploader_email}, file {filename}: {ve}", exc_info=False)
         error_code = 422
         err_status_detail = "parsing_failed" if "Gemini" in str(ve) or "JSON" in str(ve) else "extraction_failed"
         if isinstance(ve, PyPDF2.errors.PdfReadError) or "Word document" in str(ve): # CHANGED: Handle DOCX errors too
            error_code = 400

         return JSONResponse(
             status_code=error_code,
             content=ApiResponse(
                 uploader_email=uploader_email,
                 candidate_email=candidate_email_processed,
                 status=err_status_detail,
                 resume_storage_status=resume_storage_status,
                 occupation_suggestion_status=occupation_suggestion_status,
                 message=f"Failed to process resume: {ve}"
             ).dict(exclude_none=True)
         )
    except Exception as e:
        logger.exception(f"Unexpected Internal Error for {uploader_email}, file {filename}: {e}")
        return JSONResponse(
             status_code=500,
             content=ApiResponse(
                 uploader_email=uploader_email,
                 candidate_email=candidate_email_processed,
                 status="internal_error",
                 resume_storage_status=resume_storage_status,
                 occupation_suggestion_status=occupation_suggestion_status,
                 message="An unexpected internal server error occurred."
             ).dict(exclude_none=True)
        )
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as unlink_err:
                 logger.error(f"Error deleting temp file {temp_file_path}: {unlink_err}")
        if file:
             await file.close()


@app.get(
    "/api/candidates/{uploader_email}/{candidate_email}",
    response_model=CandidateDataResponse,
    responses={
        404: {"description": "Uploader or Candidate resume data not found"},
        503: {"description": "Database service unavailable"},
        500: {"description": "Internal server error"}
    },
    tags=["Candidate Data"]
)
async def get_candidate_data(
    uploader_email: str = Path(..., description="Email of the uploader."),
    candidate_email: str = Path(..., description="Email of the candidate."),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    # Verify if the current user has access to this data
    if not verify_access(current_user, uploader_email):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this data"
        )
    
    if resume_collection is None:
         logger.error("Resume collection object is None.")
         raise HTTPException(status_code=503, detail="Database service unavailable.")

    logger.info(f"Request to retrieve resume data for candidate '{candidate_email}' under uploader '{uploader_email}'.")
    uploader_email_lower = uploader_email.lower()
    candidate_email_lower = candidate_email.lower()

    try:
        uploader_doc = resume_collection.find_one(
            {"_id": uploader_email_lower, "candidates.candidate_email": candidate_email_lower},
            {"candidates.$": 1}
        )

        if uploader_doc and 'candidates' in uploader_doc and uploader_doc['candidates']:
            candidate_info = uploader_doc['candidates'][0]
            parsed_data = candidate_info.get('parsed_resume_data', {})

            try:
                validated_data = ParsedResumeDataStorage(**parsed_data)
            except Exception as pyd_err:
                 logger.error(f"Retrieved resume data for {candidate_email_lower} failed Pydantic validation: {pyd_err}")
                 raise HTTPException(status_code=500, detail="Retrieved data format error.")

            logger.info(f"Successfully found resume data for candidate '{candidate_email}' under uploader '{uploader_email}'.")
            return CandidateDataResponse(
                uploader_email=uploader_email,
                candidate_email=candidate_email,
                candidate_data=validated_data
            )
        else:
            logger.warning(f"Resume data not found for candidate '{candidate_email}' under uploader '{uploader_email}'.")
            raise HTTPException(status_code=404, detail="Candidate resume data not found for the specified uploader and candidate email.")

    except OperationFailure as op_err:
        logger.error(f"DB error retrieving resume data for candidate '{candidate_email_lower}': {op_err.details}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database operation error: {op_err.code_name}")
    except Exception as e:
        logger.exception(f"Unexpected error retrieving resume data for candidate '{candidate_email_lower}': {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


@app.put(
    "/api/candidates/{uploader_email}/{candidate_email}",
    status_code=200,
    summary="Update Candidate Resume Data (Replacement)",
    response_model=Dict[str, str],
    responses={
        200: {"description": "Candidate data updated successfully or was already up-to-date"},
        404: {"description": "Uploader or Candidate not found"},
        400: {"description": "Invalid input data (validation error or email mismatch)"},
        503: {"description": "Database service unavailable"},
        500: {"description": "Internal server error"}
    },
    tags=["Candidate Data"]
)
async def update_candidate_data(
    uploader_email: str = Path(..., description="Email of the uploader"),
    candidate_email: str = Path(..., description="Email of the candidate to update"),
    updated_data: UpdateCandidateDataRequest = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    # Verify if the current user has access to this data
    if not verify_access(current_user, uploader_email):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this data"
        )
    
    if resume_collection is None:
        logger.error("Resume collection object is None. Cannot process update.")
        raise HTTPException(status_code=503, detail="Database service unavailable.")

    logger.info(f"Request to UPDATE/REPLACE resume data for candidate '{candidate_email}' under uploader '{uploader_email}'.")
    uploader_email_lower = uploader_email.lower()
    candidate_email_lower = candidate_email.lower()

    if updated_data.email and updated_data.email.lower() != candidate_email_lower:
        logger.error(f"Update rejected: Path email '{candidate_email_lower}' does not match body email '{updated_data.email.lower()}'.")
        raise HTTPException(status_code=400, detail="Candidate email in URL path and request body must match.")
    if not updated_data.email:
         logger.error(f"Update rejected: Candidate email missing in request body for replacement.")
         raise HTTPException(status_code=400, detail="Candidate email is required in the request body for update.")

    update_payload = updated_data.dict(exclude_none=True) # exclude_none is good practice
    logger.debug(f"Update payload prepared for MongoDB: {update_payload}")

    try:
        result = resume_collection.update_one(
            {"_id": uploader_email_lower, "candidates.candidate_email": candidate_email_lower},
            {"$set": {"candidates.$.parsed_resume_data": update_payload}}
        )

        if result.matched_count == 0:
            logger.warning(f"Update failed: Uploader '{uploader_email}' or Candidate '{candidate_email}' not found.")
            raise HTTPException(status_code=404, detail="Uploader or Candidate not found.")
        elif result.modified_count >= 0:
            mod_status = "modified" if result.modified_count > 0 else "matched (no change)"
            logger.info(f"Successfully processed update request for candidate '{candidate_email}' ({mod_status}).")
            return {"message": "Candidate data update processed successfully."}

    except OperationFailure as op_err:
        logger.error(f"DB failure during update for candidate '{candidate_email}': {op_err.details}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database operation error: {op_err.code_name}")
    except Exception as e:
        logger.exception(f"Unexpected error during update for candidate '{candidate_email}': {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred during update.")


@app.get(
    "/api/occupations/{uploader_email}/{candidate_email}",
    response_model=OccupationSuggestionResponse,
    responses={
        404: {"description": "Occupation suggestions not found for this uploader/candidate"},
        503: {"description": "Database service unavailable"},
        500: {"description": "Internal server error"}
    },
    tags=["Occupation Suggestions"]
)
async def get_occupation_suggestions(
    uploader_email: str = Path(..., description="Email of the uploader."),
    candidate_email: str = Path(..., description="Email of the candidate."),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    # Verify if the current user has access to this data
    if not verify_access(current_user, uploader_email):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this data"
        )
    
    if occupation_collection is None:
         logger.error("Occupation collection object is None.")
         raise HTTPException(status_code=503, detail="Database service unavailable.")

    logger.info(f"Request to retrieve occupation suggestions for candidate '{candidate_email}' under uploader '{uploader_email}'.")
    uploader_email_lower = uploader_email.lower()
    candidate_email_lower = candidate_email.lower()
    storage_id = f"{uploader_email_lower}_{candidate_email_lower}"

    try:
        suggestion_doc = occupation_collection.find_one({"_id": storage_id})

        if suggestion_doc:
            try:
                validated_suggestions = OccupationSuggestionStorage(**suggestion_doc)
            except Exception as pyd_err:
                 logger.error(f"Retrieved occupation suggestion data for {storage_id} failed Pydantic validation: {pyd_err}")
                 raise HTTPException(status_code=500, detail="Retrieved suggestion data format error.")

            logger.info(f"Successfully found occupation suggestions for candidate '{candidate_email}' under uploader '{uploader_email}'.")
            return OccupationSuggestionResponse(
                uploader_email=uploader_email,
                candidate_email=candidate_email,
                suggestions=validated_suggestions
            )
        else:
            logger.warning(f"Occupation suggestions not found for candidate '{candidate_email}' under uploader '{uploader_email}' (ID: {storage_id}).")
            raise HTTPException(status_code=404, detail="Occupation suggestions not found for the specified uploader and candidate email.")

    except OperationFailure as op_err:
        logger.error(f"DB error retrieving suggestions for ID '{storage_id}': {op_err.details}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database operation error: {op_err.code_name}")
    except Exception as e:
        logger.exception(f"Unexpected error retrieving suggestions for ID '{storage_id}': {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


class SearchedOccupationDetail(BaseModel):
    matchedOccupation: str
    matchedANZSCO: str
    # embedding: List[float] # Removed as per your request

# Add this new endpoint function with your other FastAPI endpoints
@app.get(
    "/api/occupations/search",
    response_model=List[SearchedOccupationDetail],
    summary="Search for occupations by name",
    tags=["Occupation Suggestions"] # Or a new tag like "Occupation Search"
)
async def search_occupations_by_name(
    search_term: str = Query(..., min_length=1, description="The word or phrase to search for in occupation names.")
):
    """
    Searches for occupations where the provided search_term is found within the occupation's name (case-insensitive).
    Returns a list of matching occupations with their ANZSCO code and full name.
    """
    if not occupation_data:
        logger.warning("Occupation search attempted, but occupation_data is not loaded or is empty.")
        # Depending on requirements, you might raise an HTTPException here:
        # raise HTTPException(status_code=503, detail="Occupation data is not available for searching.")
        return []

    search_term_lower = search_term.lower()
    matched_occupations: List[SearchedOccupationDetail] = []

    for occ_item in occupation_data:
        # occ_item is expected to be a dict like:
        # {"ANZSCO": "...", "Occupation": "...", "embedding": [...]}
        # This structure is established when occupation_data is loaded.
        
        occupation_name_from_data = occ_item.get("Occupation")
        anzsco_code_from_data = occ_item.get("ANZSCO")
        
        if occupation_name_from_data and anzsco_code_from_data and search_term_lower in occupation_name_from_data.lower():
            try:
                # Create the response item using the updated Pydantic model
                # Map fields from occ_item to the fields in SearchedOccupationDetail
                matched_occupations.append(
                    SearchedOccupationDetail(
                        matchedOccupation=occupation_name_from_data,
                        matchedANZSCO=anzsco_code_from_data
                    )
                )
            except Exception as e:
                # Log if an item from occupation_data fails to conform to SearchedOccupationDetail model
                # or if "Occupation" or "ANZSCO" keys are missing.
                logger.error(f"Error processing occupation item during search: ANZSCO {anzsco_code_from_data or 'Unknown'}. Error: {e}")
                # Optionally, skip this item and continue searching
                continue
    
    if not matched_occupations:
        logger.info(f"No occupations found matching search term: '{search_term}'")
    else:
        logger.info(f"Found {len(matched_occupations)} occupations matching search term: '{search_term}'")

    return matched_occupations

class AddOccupationRequest(BaseModel):
    matchedOccupation: str = Field(..., min_length=1, description="The name of the occupation to add.")
    matchedANZSCO: str = Field(..., min_length=1, description="The ANZSCO code of the occupation to add.")

@app.post(
    "/api/occupations/{uploader_email}/{candidate_email}/manual-add",
    response_model=OccupationSuggestionResponse, # Reusing existing response model
    status_code=201, # 201 for successful creation/update leading to creation
    summary="Manually add an occupation suggestion for a candidate",
    responses={
        200: {"description": "Occupation successfully added/updated to existing suggestion list"},
        201: {"description": "Occupation added and new suggestion record created"},
        400: {"description": "Invalid input data"},
        404: {"description": "This specific combination might not be directly applicable if upsert always succeeds"},
        503: {"description": "Database service unavailable"},
        500: {"description": "Internal server error"}
    },
    tags=["Occupation Suggestions"]
)
async def add_manual_occupation_suggestion(
    uploader_email: EmailStr = Path(..., description="Email of the uploader."),
    candidate_email: EmailStr = Path(..., description="Email of the candidate."),
    payload: AddOccupationRequest = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    # Verify if the current user has access to this data
    if not verify_access(current_user, uploader_email):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this data"
        )
    
    if occupation_collection is None:
        logger.error("Occupation collection object is None. Cannot add manual suggestion.")
        raise HTTPException(status_code=503, detail="Database service unavailable.")

    uploader_email_lower = uploader_email.lower()
    candidate_email_lower = candidate_email.lower()
    storage_id = f"{uploader_email_lower}_{candidate_email_lower}"

    logger.info(f"Request to manually add occupation '{payload.matchedOccupation}' (ANZSCO: {payload.matchedANZSCO}) for candidate '{candidate_email_lower}' by uploader '{uploader_email_lower}'.")

    try:
        new_match_detail = MatchedOccupationDetail(
            matchedOccupation=payload.matchedOccupation,
            matchedANZSCO=payload.matchedANZSCO,
            similarityScore=1.0,  # Indicating a perfect, manual match/selection
            basedOnSuggestion="Manually Added by User"
        )
    except Exception as pyd_err:
        logger.error(f"Pydantic validation failed for creating MatchedOccupationDetail from payload: {pyd_err}")
        raise HTTPException(status_code=400, detail=f"Invalid occupation data provided: {pyd_err}")

    current_time_utc = datetime.utcnow().isoformat() + "Z"

    try:
        # Using find_one_and_update with upsert=True to handle both creation and update atomically.
        # $addToSet ensures the occupation is added to matched_details only if not already present.
        # $setOnInsert initializes fields only if a new document is created.
        update_operations = {
            "$set": {
                "last_updated": current_time_utc,
                "uploader_email": uploader_email_lower, # Ensure these are set/updated
                "candidate_email": candidate_email_lower
            },
            "$addToSet": {"matched_details": new_match_detail.dict(exclude_none=True)},
            "$setOnInsert": {
                # "_id": storage_id, # MongoDB handles _id creation or matches based on query
                "suggested_by_llm": [] # Initialize as empty list if new record
            }
        }
        
        # Atomically find and update (or insert if not found)
        updated_document_data = occupation_collection.find_one_and_update(
            {"_id": storage_id},
            update_operations,
            upsert=True,
            return_document=ReturnDocument.AFTER # Get the document after the update
        )

        if not updated_document_data:
            # This case should ideally not be reached if upsert=True and DB is working,
            # but as a safeguard or if ReturnDocument.AFTER wasn't supported/worked as expected.
            logger.error(f"Failed to upsert and retrieve document for {storage_id}, though find_one_and_update should have returned it.")
            raise HTTPException(status_code=500, detail="Failed to process occupation addition. Document not returned after upsert.")

        # Validate the complete updated document structure before responding
        try:
            validated_suggestions = OccupationSuggestionStorage(**updated_document_data)
        except Exception as pyd_err:
            logger.error(f"Upserted occupation suggestion data for {storage_id} failed Pydantic validation: {pyd_err}. Data: {updated_document_data}")
            raise HTTPException(status_code=500, detail="Data integrity error after updating suggestions.")

        # Determine if it was a creation or an update for logging/status code
        # One way to check if it was an insert is if 'suggested_by_llm' was just set to [] and 'matched_details' has only one item.
        # However, find_one_and_update doesn't directly return if it was an upsert vs update in a simple flag via pymongo's default result.
        # We can infer. If the document previously didn't exist, it's a create.
        # For simplicity, we'll use 200 if it previously existed and just updated, and 201 for new creation.
        # The `updated_document_data` will be the new or updated doc.
        # A more robust way to check for creation would be to query first, then update, but find_one_and_update is more atomic.
        # Let's assume status 200 for simplicity if it just "adds or updates" successfully.
        # If we want to be more specific about 201 (Created) vs 200 (OK - modified), we'd need more complex logic or
        # check the result of a simpler update_one which returns `upserted_id`.
        # For now, let's return a general success and rely on the response body.
        # The prompt asked to respond with OccupationSuggestionResponse, which find_one_and_update with AFTER gives us.

        logger.info(f"Successfully added/updated manual occupation for {storage_id}. Matched details count: {len(validated_suggestions.matched_details)}")
        
        # Check if the specific item was indeed added (if $addToSet didn't add it because it was a duplicate)
        # This check is more for confirming $addToSet behavior rather than creation status
        item_found_in_response = any(
            detail.matchedOccupation == new_match_detail.matchedOccupation and
            detail.matchedANZSCO == new_match_detail.matchedANZSCO and
            detail.basedOnSuggestion == new_match_detail.basedOnSuggestion # and score matches
            for detail in validated_suggestions.matched_details
        )

        if item_found_in_response:
             status_to_return = 200 # Or 201 if you can reliably detect creation
             # If you want to distinguish, you'd check if len(validated_suggestions.matched_details) == 1
             # and validated_suggestions.suggested_by_llm == [] for a 'fresh' manual add.
             # For now, 200 is fine as "processed".
             message_prefix = "Occupation added/updated in suggestions list."
        else:
            # This means $addToSet found an exact duplicate and didn't add it again.
            # The last_updated and other fields would still be updated.
            status_to_return = 200 
            message_prefix = "Occupation suggestion list updated (item might have already existed)."
            logger.info(f"Manual occupation for {storage_id} was likely a duplicate for $addToSet; list updated.")


        return OccupationSuggestionResponse(
            uploader_email=validated_suggestions.uploader_email,
            candidate_email=validated_suggestions.candidate_email,
            suggestions=validated_suggestions
        )
        # To be more precise with 201:
        # If you use `occupation_collection.update_one` instead of `find_one_and_update`:
        # result = occupation_collection.update_one({"_id": storage_id}, update_operations, upsert=True)
        # if result.upserted_id:
        #     status_code_to_use = 201
        # else:
        #     status_code_to_use = 200
        # Then you'd need to fetch the document again to return it.
        # `find_one_and_update` is more efficient if you need the document back.

    except OperationFailure as op_err:
        logger.error(f"MongoDB operation failed adding manual suggestion for {storage_id}: {op_err.details}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database operation error: {op_err.code_name}")
    except HTTPException as http_exc: # Re-raise HTTPException
        raise http_exc
    except Exception as e:
        logger.exception(f"Unexpected error adding manual suggestion for {storage_id}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected internal server error occurred.")


class EligibilityListsModel(BaseModel):
    MLTSSL: Optional[int] = None
    STSOL: Optional[int] = None
    ROL: Optional[int] = None
    CSOL: Optional[int] = None

class VisaSubclassesModel(BaseModel):
    SC_482_SID: Optional[int] = Field(None, alias="SC_482_SID")
    SC_494: Optional[int] = Field(None, alias="SC_494")
    ENS_SC_186: Optional[int] = Field(None, alias="ENS_SC_186")
    SC_189: Optional[int] = Field(None, alias="SC_189")
    SC_190: Optional[int] = Field(None, alias="SC_190")
    SC_491_State: Optional[int] = Field(None, alias="SC_491_State")
    SC_491_Family: Optional[int] = Field(None, alias="SC_491_Family")
    SC_485: Optional[int] = Field(None, alias="SC_485")
    SC_407: Optional[int] = Field(None, alias="SC_407")

class ACTVisaModel(BaseModel):
    visa_190: Optional[int] = Field(None, alias="190")
    visa_491: Optional[int] = Field(None, alias="491")

class VICVisaModel(BaseModel):
    visa_190: Optional[int] = Field(None, alias="190")
    visa_491: Optional[int] = Field(None, alias="491")

class SAVisaModel(BaseModel):
    visa_190: Optional[int] = Field(None, alias="190")
    visa_491_SA_Grad: Optional[int] = Field(None, alias="491_SA_Grad")
    visa_491_Skilled: Optional[int] = Field(None, alias="491_Skilled")
    visa_491_Outer: Optional[int] = Field(None, alias="491_Outer")
    visa_491_Offshore: Optional[int] = Field(None, alias="491_Offshore")

class WAVisaModel(BaseModel):
    visa_190_WS1: Optional[int] = Field(None, alias="190_WS1")
    visa_190_WS2: Optional[int] = Field(None, alias="190_WS2") # Assuming int or null based on example; 0 is a value
    visa_190_Grad: Optional[int] = Field(None, alias="190_Grad") # Assuming int or null

class QLDVisaModel(BaseModel):
    visa_190: Optional[int] = Field(None, alias="190")
    visa_491_Onshore: Optional[int] = Field(None, alias="491_Onshore")
    visa_491_Offshore: Optional[int] = Field(None, alias="491_Offshore")

class TASVisaModel(BaseModel):
    visa_190: Optional[int] = Field(None, alias="190")
    visa_491: Optional[int] = Field(None, alias="491")

class NTVisaModel(BaseModel):
    visa_190: Optional[int] = Field(None, alias="190")
    visa_491_Onshore: Optional[int] = Field(None, alias="491_Onshore")
    visa_491_Offshore: Optional[int] = Field(None, alias="491_Offshore")

class StateOccupationDetailModel(BaseModel):
    ACT: Optional[ACTVisaModel] = None
    VIC: Optional[VICVisaModel] = None
    SA: Optional[SAVisaModel] = None
    WA: Optional[WAVisaModel] = None
    QLD: Optional[QLDVisaModel] = None
    TAS: Optional[TASVisaModel] = None
    NT: Optional[NTVisaModel] = None

class DAMAModel(BaseModel):
    Adelaide: Optional[int] = None
    Regional_SA: Optional[int] = Field(None, alias="Regional_SA")

class ANZSCOFullDetailResponse(BaseModel):
    ANZSCO: str
    Occupation: str
    Assessing_Authority: Optional[str] = None # Handles "nan" as a string, or null as None
    eligibility_lists: EligibilityListsModel
    visaSubclasses: VisaSubclassesModel
    State: StateOccupationDetailModel
    DAMA: DAMAModel

    class Config:
        # For Pydantic v2: model_config = {"populate_by_name": True}
        # For Pydantic v1: populate_by_name = True (or by_alias in .dict())
        # This helps with field names like "190" vs "visa_190" if needed, but Field(alias=...) is more direct.
        pass
# --- End of Pydantic Models for ANZSCO Detail Response ---

full_occupation_json_list: List[Dict[str, Any]] = []

try:
    with open(OCCUPATIONS_EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
        # The variable 'all_data' in your original code loads the full JSON content.
        # Let's use a slightly different local name here to avoid any potential confusion
        # if 'all_data' was intended to be seen as global by you.
        loaded_json_content_local = json.load(f)

        # Populate the NEW global list with the complete data from the file.
        # This list will be used by the new /occupation/{anzsco_code} endpoint.
        # No 'global' keyword is strictly needed here if this is at the module's top level scope,
        # but if this logic were in a function, 'global full_occupation_json_list' would be required.
        full_occupation_json_list.clear() # Clear if it might be reloaded (good practice)
        full_occupation_json_list.extend(loaded_json_content_local) # Assign the loaded list

        logger.info(f"Loaded {len(full_occupation_json_list)} full occupation records from {OCCUPATIONS_EMBEDDINGS_FILE}.")

        # Now, populate the existing 'occupation_data' list for embedding-based matching
        # This part of your logic remains, but it iterates over 'loaded_json_content_local'.
        occupation_data.clear() # Clear before repopulating
        for item in loaded_json_content_local: # Iterate the full data we just loaded
            if (item.get("Occupation") and
                item.get("ANZSCO") and
                isinstance(item.get("occupation_embedding"), list) and # Check original key
                len(item["occupation_embedding"]) > 0):
                occupation_data.append({ # Appending to the global 'occupation_data'
                    "ANZSCO": item["ANZSCO"],
                    "Occupation": item["Occupation"],
                    "embedding": item["occupation_embedding"] # Storing under 'embedding' key
                })
        logger.info(f"Processed {len(occupation_data)} occupations with embeddings for matching.")
        
        if not occupation_data: # This check is for the embedding-based matching
            logger.warning(f"Warning: No valid occupations with embeddings found in {OCCUPATIONS_EMBEDDINGS_FILE}. Matching may be limited.")

except FileNotFoundError:
    logger.error(f"Error: Occupations file '{OCCUPATIONS_EMBEDDINGS_FILE}' not found. Occupation data and matching will be disabled.")
    full_occupation_json_list.clear() # Ensure new list is also cleared on error
    occupation_data.clear() # Ensure existing list is cleared
except json.JSONDecodeError:
    logger.error(f"Error: Could not decode JSON from '{OCCUPATIONS_EMBEDDINGS_FILE}'. Occupation data and matching will be disabled.")
    full_occupation_json_list.clear()
    occupation_data.clear()
except Exception as e:
    logger.exception(f"Error loading or processing occupation data from {OCCUPATIONS_EMBEDDINGS_FILE}: {e}")
    full_occupation_json_list.clear()
    occupation_data.clear()

@app.get(
    "/occupation/{anzsco_code}",
    response_model=ANZSCOFullDetailResponse,
    summary="Get full details for a specific ANZSCO occupation",
    tags=["Occupation Details"], # You can create a new tag
    responses={
        200: {"description": "Successfully retrieved occupation details"},
        404: {"description": "Occupation with the specified ANZSCO code not found"},
        500: {"description": "Data format error or internal server error"},
        503: {"description": "Occupation data is currently unavailable"}
    }
)
async def get_occupation_details_by_anzsco(
    anzsco_code: str = Path(..., description="The ANZSCO code of the occupation (e.g., '139914').")
):
    """
    Retrieves detailed information for a given ANZSCO code, including eligibility lists,
    visa subclasses, state nomination options, and DAMA agreements.
    The 'occupation_embedding' field is excluded from the response by the Pydantic model.
    """
    if not full_occupation_json_list: # Check the new global list
        logger.error("Full occupation data (full_occupation_json_list) is not loaded. Cannot serve ANZSCO details.")
        raise HTTPException(status_code=503, detail="Occupation data is currently unavailable.")

    # Find the occupation by iterating through the list
    # This is less efficient than a dict lookup but uses the simple list structure
    found_occupation_data = None
    for occupation_item in full_occupation_json_list:
        if occupation_item.get("ANZSCO") == anzsco_code:
            found_occupation_data = occupation_item
            break 

    if not found_occupation_data:
        logger.warning(f"Request for non-existent ANZSCO code: {anzsco_code}")
        raise HTTPException(status_code=404, detail=f"Occupation with ANZSCO code '{anzsco_code}' not found.")

    try:
        # Pydantic will parse the raw dictionary (found_occupation_data).
        # Fields not defined in ANZSCOFullDetailResponse (like 'occupation_embedding') will be ignored.
        # It will also validate the structure and types based on your models.
        response_data = ANZSCOFullDetailResponse(**found_occupation_data)
        logger.info(f"Successfully retrieved and validated details for ANZSCO code: {anzsco_code}")
        return response_data
    except ValidationError as e:
        logger.error(f"Data validation error for ANZSCO code {anzsco_code}. Data from file might be malformed. Error: {e.errors()}")
        # For debugging, you might want to log the problematic data, but be careful in production:
        # logger.debug(f"Problematic data for ANZSCO {anzsco_code}: {found_occupation_data}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal data format error for ANZSCO code '{anzsco_code}'. Please check server logs."
        )
    except Exception as e:
        logger.exception(f"An unexpected error occurred while processing ANZSCO code {anzsco_code}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected server error occurred for ANZSCO code '{anzsco_code}'."
        )
    
temp_dir = tempfile.mkdtemp()
# Store document in MongoDB
visa_collection = db['visa_rules']

@app.post("/upload-visa-rules/")
async def upload_file(
    file: UploadFile = File(...),
    visa_subclass: str = Form(...),
    state: Optional[str] = Form(None),
    current_user: Dict[str, Any] = Depends(get_current_user) # This dependency must be here
):
    """Process the uploaded Word file, extract text, and store in MongoDB"""

    # --- ADD THIS ADMIN CHECK ---
    user_role = current_user.get("role", "user") # Get the user's role, default to 'user' if not set
    if user_role != "admin":
        logger.warning(f"Access denied for user '{current_user.get('email')}': Tried to upload visa rules but role is '{user_role}' (requires 'admin').")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to upload visa rules. Only administrators can perform this action."
        )
    # --- END ADMIN CHECK ---
    
    # Validate file type
    if not file.filename.endswith('.docx'):
        raise HTTPException(status_code=400, detail="Only .docx files are supported")
    
    # Save the file temporarily
    file_path = os.path.join(temp_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    try:
        # Extract text from the Word document
        with open(file_path, "rb") as docx_file:
            result = mammoth.extract_raw_text(docx_file)
            text = result.value
        
        # Get document name (filename without extension)
        document_name = os.path.splitext(file.filename)[0]

        visa_prompt = f"""Please carefully review the document text provided below and identify ALL important pieces of information that are typically essential for evaluating this type of visa application.",
                    f"Present your findings as a comprehensive but concise itemized list. If specific information is missing but expected, note that as well.",
                    f"\n\n--- Document Text Start ---\n{text}\n--- Document Text End ---"""

        response = gemini_client.models.generate_content(
            model=f"models/{gemini_model_name}",
            contents=visa_prompt,
            # generation_config=genai.types.GenerationConfig(
            #     # response_mime_type="application/json" # Enable if model supports reliable JSON mode
            #     temperature=0.2 # Lower temperature for more deterministic structured output
            # )
        )
        initial_analysis = response.text
        
        # Create document for MongoDB
        document = {
            "document_name": document_name,
            "visa_subclass": visa_subclass,
            "text": text,
            "initial_analysis": initial_analysis
        }
        
        # Add state if provided
        if state:
            document["state"] = state
        
        # Insert into MongoDB
        result = visa_collection.insert_one(document)
        
        # Clean up temp file
        os.remove(file_path)
        
        return {
            "status": "success",
            "message": "Document processed successfully",
            "document_id": str(result.inserted_id),
            "document_name": document_name,
            "initial_analysis": initial_analysis,
            "visa_subclass": visa_subclass,
            "state": state,
            "text_length": len(text),
            "text_preview": text[:200] + "..." if len(text) > 200 else text
        }
        
    except Exception as e:
        # Clean up temp file in case of error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
@app.get("/documents/")
async def list_documents(current_user: Dict[str, Any] = Depends(get_current_user) # This dependency must be here
):
    """Process the uploaded Word file, extract text, and store in MongoDB"""

    # --- ADD THIS ADMIN CHECK ---
    user_role = current_user.get("role", "user") # Get the user's role, default to 'user' if not set
    if user_role != "admin":
        logger.warning(f"Access denied for user '{current_user.get('email')}': Tried to upload visa rules but role is '{user_role}' (requires 'admin').")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to upload visa rules. Only administrators can perform this action."
        )
    # --- END ADMIN CHECK ---
    """List all documents in the visa_rules collection"""
    documents = []
    for doc in visa_collection.find({}, {"text": 1}):  # Exclude text field for brevity
        doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
        documents.append(doc)
    
    return {"documents": documents}

class UserInfoFlexible(BaseModel): # Renamed to avoid conflict if UserInfo is used elsewhere with strict schema
    uploader_email: str
    candidate_email: EmailStr
    candidate_data: Dict[str, Any] # Candidate data will be a flexible dictionary
    questions_candidate_answered: Dict[str, Any] # Additional info will be a flexible dictionary

class SkillVisaEvaluationResponseFlexible(BaseModel): # Renamed for clarity
    user_info: List[UserInfoFlexible]
    visa_rules: Dict[str, Any] # Visa rules will be a flexible dictionary
    nominated_occupation: str
    visa_subclass: str

@app.get(
    "/api/evaluate/{uploader_email}/{candidate_email}/visa/common-state-questions", # Modified path for clarity
    response_model=Any, # FastAPI endpoint will now return the raw JSON from webhook (parsed to Python object)
    tags=["Visa Evaluation"],
    summary="Triggers an n8n webhook and returns its raw JSON response",
    description="Combines candidate data with visa rules, sends it to an n8n webhook, and returns the entire JSON response received from the webhook as-is.",
    responses={
        200: {"description": "Successfully retrieved raw response from the webhook"},
        404: {"description": "Candidate, Uploader, or Visa Rule (for internal payload) not found"},
        500: {"description": "Internal server error"},
        502: {"description": "Bad Gateway - Error communicating with or non-JSON response from webhook service"},
        503: {"description": "Database service unavailable"}
    }
)
async def get_raw_webhook_response( # Renamed function for clarity
    uploader_email: str = Path(..., description="Email of the uploader."),
    candidate_email: str = Path(..., description="Email of the candidate."),
    nominated_occupation: str = Query(..., description="The occupation nominated by the candidate."),
    visa_subclass: str = Query(..., description="The visa subclass the candidate is applying for."),
    state: Optional[str] = Query(None, description="The state for state-specific visa rules (e.g., 'Western Australia'). If provided, rules specific to this state and visa subclass will be fetched."),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    # Verify if the current user has access to this data
    if not verify_access(current_user, uploader_email):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this data"
        )
    
    # N8N_WEBHOOK_URL = "https://mwxeosxtdobcbzsxku.app.n8n.cloud/webhook-test/228754dd-ab65-4559-8654-34c2f6f08f14"
    # N8N_WEBHOOK_URL = "https://zjpxpyanroshiskdre.app.n8n.cloud/webhook/228754dd-ab65-4559-8654-34c2f6f08f14"
    # N8N_WEBHOOK_URL = "https://zjpxpyanroshiskdre.app.n8n.cloud/webhook-test/228754dd-ab65-4559-8654-34c2f6f08f43"
    N8N_WEBHOOK_URL = "https://zjpxpyanroshiskdre.app.n8n.cloud/webhook/228754dd-ab65-4559-8654-34c2f6f08f43"
    logger.info(f"Request to get raw webhook response for candidate '{candidate_email}' (uploader: '{uploader_email}') via n8n.")

    # --- 1. Prepare the payload for the webhook (same logic as before) ---
    if resume_collection is None:
        logger.error("resume_collection is not initialized.")
        raise HTTPException(status_code=503, detail="Database service (resume_collection) unavailable.")
    if visa_collection is None:
        logger.error("visa_collection is not initialized.")
        raise HTTPException(status_code=503, detail="Database service (visa_collection) unavailable.")

    uploader_email_lower = uploader_email.lower()
    candidate_email_lower = candidate_email.lower()

    found_user_info_object: Optional[UserInfoFlexible] = None
    try:
        uploader_doc = resume_collection.find_one(
            {"_id": uploader_email_lower, "candidates.candidate_email": candidate_email_lower},
            {"candidates.$": 1}
        )
        interactions_collection = db['additional_info']
        try:
            qa_doc = interactions_collection.find_one(
                {"uploader_email": uploader_email_lower, "candidate_email": candidate_email_lower},
                {"additional_info.$": 1}
            )
        except:
            print('no quetions asked')
        if not uploader_doc or 'candidates' not in uploader_doc or not uploader_doc['candidates']:
            raise HTTPException(status_code=404, detail="Candidate resume data not found.")
        candidate_entry = uploader_doc['candidates'][0]
        raw_candidate_data = candidate_entry
        if qa_doc:
            candidate_answers = qa_doc['additional_info']
            raw_candidate_answers = candidate_answers
        else:
            raw_candidate_answers = {}
        # raw_additional_info = candidate_entry.get('additional_info')
        if not raw_candidate_data or not isinstance(raw_candidate_data, dict):
            raise HTTPException(status_code=404, detail="Candidate's parsed resume data is missing or not in expected dict format.")
        found_user_info_object = UserInfoFlexible(
            uploader_email=uploader_email, candidate_email=candidate_email, candidate_data=raw_candidate_data, questions_candidate_answered=raw_candidate_answers
        )
    except OperationFailure as op_err:
        logger.error(f"DB error fetching candidate data: {op_err.details}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"DB error fetching candidate data: {op_err.code_name}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error fetching candidate data: {e}")
        raise HTTPException(status_code=500, detail="Internal error fetching candidate data.")

    found_visa_rules_dict: Optional[Dict[str, Any]] = None
    try:
        visa_rules_query: Dict[str, Any] = {"visa_subclass": '190'}
        if state: visa_rules_query["state"] = 'common'
        visa_rule_doc_raw = visa_collection.find_one(visa_rules_query)
        if not visa_rule_doc_raw:
            detail_msg = f"Visa rules not found for subclass '{visa_subclass}'"
            if state: detail_msg += f" and state '{state}'"
            raise HTTPException(status_code=404, detail=detail_msg + ".")
        if "_id" in visa_rule_doc_raw and not isinstance(visa_rule_doc_raw["_id"], str):
            visa_rule_doc_raw["_id"] = str(visa_rule_doc_raw["_id"])
        found_visa_rules_dict = visa_rule_doc_raw
    except OperationFailure as op_err:
        logger.error(f"DB error fetching visa rules: {op_err.details}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"DB error fetching visa rules: {op_err.code_name}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error fetching visa rules: {e}")
        raise HTTPException(status_code=500, detail="Internal error fetching visa rules.")

    if found_user_info_object is None or found_visa_rules_dict is None:
        raise HTTPException(status_code=500, detail="Internal error: data fetching for webhook payload incomplete.")

    webhook_payload = SkillVisaEvaluationResponseFlexible(
        user_info=[found_user_info_object],
        visa_rules=found_visa_rules_dict,
        nominated_occupation=nominated_occupation,
        visa_subclass=visa_subclass
    )
    logger.info(f"Webhook payload successfully prepared for candidate '{candidate_email}'.")

    # --- 2. Call the n8n Webhook and return its response as-is ---
    try:
        logger.info(f"Sending data to n8n webhook: {N8N_WEBHOOK_URL} and awaiting its raw response.")
        webhook_response = requests.post(N8N_WEBHOOK_URL, json=webhook_payload.dict(), timeout=50)
        webhook_response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        
        # Parse the JSON response from the webhook
        raw_webhook_data = webhook_response.json() 
        
        logger.info(f"Successfully received and parsed response from n8n webhook. Status: {webhook_response.status_code}.")
        return raw_webhook_data # Return the parsed JSON object directly

    except requests.exceptions.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from webhook: {e}. Response text: {webhook_response.text if 'webhook_response' in locals() else 'N/A'}")
        raise HTTPException(status_code=502, detail="Webhook service returned non-JSON response.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling n8n webhook: {e}")
        if e.response is not None:
            logger.error(f"Webhook response status: {e.response.status_code}, Text: {e.response.text}")
            if 400 <= e.response.status_code < 600:
                 raise HTTPException(status_code=502, detail=f"Webhook service error: {e.response.status_code} - {e.response.reason}")
        raise HTTPException(status_code=502, detail="Failed to communicate with webhook service.")
    except Exception as e_general: # Catch any other unexpected errors
        logger.exception(f"An unexpected error occurred during or after webhook call: {e_general}")
        raise HTTPException(status_code=500, detail="Internal server error processing webhook interaction.")

@app.get(
    "/api/evaluate/{uploader_email}/{candidate_email}/visa/webhook-response", # Modified path for clarity
    response_model=Any, # FastAPI endpoint will now return the raw JSON from webhook (parsed to Python object)
    tags=["Visa Evaluation"],
    summary="Triggers an n8n webhook and returns its raw JSON response",
    description="Combines candidate data with visa rules, sends it to an n8n webhook, and returns the entire JSON response received from the webhook as-is.",
    responses={
        200: {"description": "Successfully retrieved raw response from the webhook"},
        404: {"description": "Candidate, Uploader, or Visa Rule (for internal payload) not found"},
        500: {"description": "Internal server error"},
        502: {"description": "Bad Gateway - Error communicating with or non-JSON response from webhook service"},
        503: {"description": "Database service unavailable"}
    }
)
async def get_raw_webhook_response( # Renamed function for clarity
    uploader_email: str = Path(..., description="Email of the uploader."),
    candidate_email: str = Path(..., description="Email of the candidate."),
    nominated_occupation: str = Query(..., description="The occupation nominated by the candidate."),
    visa_subclass: str = Query(..., description="The visa subclass the candidate is applying for."),
    state: Optional[str] = Query(None, description="The state for state-specific visa rules (e.g., 'Western Australia'). If provided, rules specific to this state and visa subclass will be fetched."),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    # Verify if the current user has access to this data
    if not verify_access(current_user, uploader_email):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this data"
        )
    
    # N8N_WEBHOOK_URL = "https://mwxeosxtdobcbzsxku.app.n8n.cloud/webhook-test/228754dd-ab65-4559-8654-34c2f6f08f14"
    # N8N_WEBHOOK_URL = "https://zjpxpyanroshiskdre.app.n8n.cloud/webhook/228754dd-ab65-4559-8654-34c2f6f08f14"
    # N8N_WEBHOOK_URL = "https://zjpxpyanroshiskdre.app.n8n.cloud/webhook-test/228754dd-ab65-4559-8654-34c2f6f08f21"
    N8N_WEBHOOK_URL = "https://zjpxpyanroshiskdre.app.n8n.cloud/webhook/228754dd-ab65-4559-8654-34c2f6f08f21"
    logger.info(f"Request to get raw webhook response for candidate '{candidate_email}' (uploader: '{uploader_email}') via n8n.")

    # --- 1. Prepare the payload for the webhook (same logic as before) ---
    if resume_collection is None:
        logger.error("resume_collection is not initialized.")
        raise HTTPException(status_code=503, detail="Database service (resume_collection) unavailable.")
    if visa_collection is None:
        logger.error("visa_collection is not initialized.")
        raise HTTPException(status_code=503, detail="Database service (visa_collection) unavailable.")

    uploader_email_lower = uploader_email.lower()
    candidate_email_lower = candidate_email.lower()

    found_user_info_object: Optional[UserInfoFlexible] = None
    try:
        uploader_doc = resume_collection.find_one(
            {"_id": uploader_email_lower, "candidates.candidate_email": candidate_email_lower},
            {"candidates.$": 1}
        )
        interactions_collection = db['additional_info']
        try:
            qa_doc = interactions_collection.find_one(
                {"uploader_email": uploader_email_lower, "candidate_email": candidate_email_lower},
                {"additional_info.$": 1}
            )
        except:
            print('no quetions asked')
        if not uploader_doc or 'candidates' not in uploader_doc or not uploader_doc['candidates']:
            raise HTTPException(status_code=404, detail="Candidate resume data not found.")
        candidate_entry = uploader_doc['candidates'][0]
        raw_candidate_data = candidate_entry
        if qa_doc:
            candidate_answers = qa_doc['additional_info']
            raw_candidate_answers = candidate_answers
        else:
            raw_candidate_answers = {}
        # raw_additional_info = candidate_entry.get('additional_info')
        if not raw_candidate_data or not isinstance(raw_candidate_data, dict):
            raise HTTPException(status_code=404, detail="Candidate's parsed resume data is missing or not in expected dict format.")
        found_user_info_object = UserInfoFlexible(
            uploader_email=uploader_email, candidate_email=candidate_email, candidate_data=raw_candidate_data, questions_candidate_answered=raw_candidate_answers
        )
    except OperationFailure as op_err:
        logger.error(f"DB error fetching candidate data: {op_err.details}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"DB error fetching candidate data: {op_err.code_name}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error fetching candidate data: {e}")
        raise HTTPException(status_code=500, detail="Internal error fetching candidate data.")

    found_visa_rules_dict: Optional[Dict[str, Any]] = None
    try:
        visa_rules_query: Dict[str, Any] = {"visa_subclass": visa_subclass}
        if state: visa_rules_query["state"] = state
        visa_rule_doc_raw = visa_collection.find_one(visa_rules_query)
        if not visa_rule_doc_raw:
            detail_msg = f"Visa rules not found for subclass '{visa_subclass}'"
            if state: detail_msg += f" and state '{state}'"
            raise HTTPException(status_code=404, detail=detail_msg + ".")
        if "_id" in visa_rule_doc_raw and not isinstance(visa_rule_doc_raw["_id"], str):
            visa_rule_doc_raw["_id"] = str(visa_rule_doc_raw["_id"])
        found_visa_rules_dict = visa_rule_doc_raw
    except OperationFailure as op_err:
        logger.error(f"DB error fetching visa rules: {op_err.details}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"DB error fetching visa rules: {op_err.code_name}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error fetching visa rules: {e}")
        raise HTTPException(status_code=500, detail="Internal error fetching visa rules.")

    if found_user_info_object is None or found_visa_rules_dict is None:
        raise HTTPException(status_code=500, detail="Internal error: data fetching for webhook payload incomplete.")

    webhook_payload = SkillVisaEvaluationResponseFlexible(
        user_info=[found_user_info_object],
        visa_rules=found_visa_rules_dict,
        nominated_occupation=nominated_occupation,
        visa_subclass=visa_subclass
    )
    logger.info(f"Webhook payload successfully prepared for candidate '{candidate_email}'.")

    # --- 2. Call the n8n Webhook and return its response as-is ---
    try:
        logger.info(f"Sending data to n8n webhook: {N8N_WEBHOOK_URL} and awaiting its raw response.")
        webhook_response = requests.post(N8N_WEBHOOK_URL, json=webhook_payload.dict(), timeout=50)
        webhook_response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        
        # Parse the JSON response from the webhook
        raw_webhook_data = webhook_response.json() 
        
        logger.info(f"Successfully received and parsed response from n8n webhook. Status: {webhook_response.status_code}.")
        return raw_webhook_data # Return the parsed JSON object directly

    except requests.exceptions.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from webhook: {e}. Response text: {webhook_response.text if 'webhook_response' in locals() else 'N/A'}")
        raise HTTPException(status_code=502, detail="Webhook service returned non-JSON response.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling n8n webhook: {e}")
        if e.response is not None:
            logger.error(f"Webhook response status: {e.response.status_code}, Text: {e.response.text}")
            if 400 <= e.response.status_code < 600:
                 raise HTTPException(status_code=502, detail=f"Webhook service error: {e.response.status_code} - {e.response.reason}")
        raise HTTPException(status_code=502, detail="Failed to communicate with webhook service.")
    except Exception as e_general: # Catch any other unexpected errors
        logger.exception(f"An unexpected error occurred during or after webhook call: {e_general}")
        raise HTTPException(status_code=500, detail="Internal server error processing webhook interaction.")

class ProcessWebhookResponse(BaseModel):
    uploader_email: str
    candidate_email: EmailStr
    status: str
    message: str
    stored_data_field_name: str = Field(..., description="The name of the field where data is stored in the new collection.")
    updated_information_preview: Dict[str, Any] = Field(..., description="A preview of the merged data that was stored.")

@app.post( # Or @app.post
    "/api/candidates/{uploader_email}/{candidate_email}/process-webhook-data", # Path unchanged as per instruction
    response_model=ProcessWebhookResponse,
    status_code=200,
    tags=["Candidate Data"],
    summary="Stores and merges custom data for a candidate.",
    description="Takes a custom JSON payload. This JSON object is stored in a dedicated collection "
                "If data for this candidate already exists in that collection, the new payload is merged into "
                "the existing data: if a key from the new data already exists, its value becomes a list "
                "(or has the new value appended if already a list), accumulating all values submitted for that key. "
                "This ensures a historical aggregation of all submitted data points. "
                "The existence of the uploader and candidate is first validated against the main resume data.",
    responses={
        200: {"description": "Custom data stored and merged successfully."},
        400: {"description": "Invalid input payload."},
        404: {"description": "Uploader or Candidate not found in the resume database."},
        500: {"description": "Internal server error during processing or DB operation."},
        503: {"description": "Database service unavailable."}
    }
)
async def process_and_store_webhook_data( # Renamed from original to avoid conflict if running side-by-side
    uploader_email: str = Path(..., description="Email of the uploader."),
    candidate_email: EmailStr = Path(..., description="Email of the candidate."),
    # Parameter name `webhook_input_payload` kept as per original function signature
    webhook_input_payload: Dict[str, Any] = Body(..., example={"question_id": "Q123", "answer": "Candidate's response.", "score": 85}),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    # Verify if the current user has access to this data
    if not verify_access(current_user, uploader_email):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this data"
        )
    """
    Stores custom data payload into a dedicated collection for the candidate,
    merging it with existing data using a specific aggregation logic.
    Validates uploader/candidate existence against the resume collection first.
    """
    interactions_collection = db['additional_info']
    AGGREGATED_DATA_FIELD_NAME = "additional_info"
    logger.info(f"Request to store and merge custom data for candidate '{candidate_email}' (uploader: '{uploader_email}').")

    if resume_collection is None or interactions_collection is None:
        logger.error("Database collection object(s) are None. Cannot process request.")
        raise HTTPException(status_code=503, detail="Database service unavailable.")

    uploader_email_lower = uploader_email.lower()
    candidate_email_lower = str(candidate_email).lower() # Ensure EmailStr is converted to str for DB query consistency
    
    data_to_store = webhook_input_payload # This is the data previously expected from the webhook

    # 1. Validate Uploader and Candidate existence in resume_collection (as per original logic)
    try:
        uploader_doc = resume_collection.find_one(
            {"_id": uploader_email_lower, "candidates.candidate_email": candidate_email_lower},
            {"_id": 1} # We only need to check for existence, no need to pull candidate data
        )
        if not uploader_doc:
            logger.warning(f"Candidate '{candidate_email_lower}' not found under uploader '{uploader_email_lower}' for storing custom data.")
            raise HTTPException(status_code=404, detail="Uploader or Candidate not found in the resume database.")
    except OperationFailure as op_err:
        logger.error(f"MongoDB operation failed during validation for candidate '{candidate_email_lower}': {op_err.details}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database validation error: {op_err.code_name}")
    except Exception as e:
        logger.exception(f"Unexpected error during uploader/candidate validation for '{candidate_email_lower}': {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during validation.")

    # 2. Fetch existing aggregated data from the new interactions_collection
    existing_aggregated_data: Dict[str, Any] = {}
    try:
        interaction_doc = interactions_collection.find_one(
            {"uploader_email": uploader_email_lower, "candidate_email": candidate_email_lower}
        )
        if interaction_doc and isinstance(interaction_doc.get(AGGREGATED_DATA_FIELD_NAME), dict):
            existing_aggregated_data = interaction_doc[AGGREGATED_DATA_FIELD_NAME]
            logger.info(f"Found existing interaction data for candidate '{candidate_email_lower}'.")
        else:
            logger.info(f"No existing interaction data found for candidate '{candidate_email_lower}'. A new record will be created/initialized.")

    except OperationFailure as op_err:
        logger.error(f"MongoDB operation failed fetching existing interaction data for '{candidate_email_lower}': {op_err.details}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error fetching interaction data: {op_err.code_name}")
    except Exception as e:
        logger.exception(f"Error fetching existing interaction data for candidate '{candidate_email_lower}': {e}")
        raise HTTPException(status_code=500, detail="Internal error fetching interaction data.")

    # 3. Prepare the merged object using the specified logic
    # Deepcopy to avoid modifying the fetched data directly if it's mutable
    merged_data = copy.deepcopy(existing_aggregated_data)

    for key, new_value in data_to_store.items():
        if key in merged_data:
            current_value = merged_data[key]
            if isinstance(current_value, list):
                current_value.append(new_value)
            else:
                merged_data[key] = [current_value, new_value]
        else:
            merged_data[key] = new_value
    
    logger.info(f"Custom data prepared for storage/merging for candidate '{candidate_email_lower}'.")

    # 4. Store the merged object in MongoDB (interactions_collection) using upsert
    try:
        current_time = datetime.now(timezone.utc)
        update_result = interactions_collection.update_one(
            {"uploader_email": uploader_email_lower, "candidate_email": candidate_email_lower},
            {
                "$set": {
                    AGGREGATED_DATA_FIELD_NAME: merged_data,
                    "uploader_email": uploader_email_lower, # Ensure these are set on update too
                    "candidate_email": candidate_email_lower,
                    "last_updated_at": current_time
                },
                "$setOnInsert": {
                    # uploader_email and candidate_email are already in $set,
                    # but $setOnInsert guarantees they are only set on creation if not in $set.
                    # However, to ensure they are always present, $set is better.
                    # "uploader_email": uploader_email_lower,
                    # "candidate_email": candidate_email_lower,
                    "created_at": current_time
                }
            },
            upsert=True
        )
        
        status_message = "Custom data stored and merged successfully."
        if update_result.upserted_id:
            status_message = "Custom data stored successfully (new record created)."
            logger.info(f"New interaction record created for candidate '{candidate_email_lower}' with ID: {update_result.upserted_id}.")
        elif update_result.modified_count > 0:
            status_message = "Custom data merged and updated successfully."
            logger.info(f"Interaction data updated for candidate '{candidate_email_lower}'.")
        elif update_result.matched_count > 0:
            status_message = "Submitted data was identical to existing aggregated data; no change made."
            logger.info(f"Submitted data for candidate '{candidate_email_lower}' resulted in no change to stored data.")
        else:
            # This case should ideally not be reached if upsert=True and no error.
            # If reached, it implies something unexpected.
            logger.error(f"Interaction data update for '{candidate_email_lower}' reported no match, no modification, and no upsert. Result: {update_result.raw_result}")
            status_message = "Data submission processed, but status unclear (no change, no new record)."


        return ProcessWebhookResponse(
            uploader_email=uploader_email,
            candidate_email=candidate_email, # Original EmailStr from input
            status="success",
            message=status_message,
            stored_data_field_name=AGGREGATED_DATA_FIELD_NAME,
            updated_information_preview=merged_data
        )

    except OperationFailure as op_err:
        logger.error(f"MongoDB operation failed storing custom data for candidate '{candidate_email_lower}': {op_err.details}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database operation error: {op_err.code_name}")
    except Exception as e:
        logger.exception(f"Unexpected error storing custom data for candidate '{candidate_email_lower}': {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred during data storage.")

class DetailEvaluationItemSchema(BaseModel):
    criteria: str
    short_explanation: str
    score: int

class CalculatedPointsBodySchema(BaseModel): # This schema represents the entire request body
    detail_evaluation: List[DetailEvaluationItemSchema]
    total_score: int

@app.get(
    "/api/evaluate/{uploader_email}/{candidate_email}/visa/all-state-eligibility", # Modified path for clarity
    response_model=Any, # FastAPI endpoint will now return the raw JSON from webhook (parsed to Python object)
    tags=["Visa Evaluation"],
    summary="Triggers an n8n webhook and returns its raw JSON response",
    description="Combines candidate data with visa rules, sends it to an n8n webhook, and returns the entire JSON response received from the webhook as-is.",
    responses={
        200: {"description": "Successfully retrieved raw response from the webhook"},
        404: {"description": "Candidate, Uploader, or Visa Rule (for internal payload) not found"},
        500: {"description": "Internal server error"},
        502: {"description": "Bad Gateway - Error communicating with or non-JSON response from webhook service"},
        503: {"description": "Database service unavailable"}
    }
)
async def get_state_eligibility(
    uploader_email: str = Path(..., description="Email of the uploader."),
    candidate_email: str = Path(..., description="Email of the candidate."),
    nominated_occupation: str = Query(..., description="The occupation nominated by the candidate."),
    visa_subclass: str = Query(..., description="The visa subclass the candidate is applying for."),
    state: Optional[str] = Query('ALL', description="The state for state-specific visa rules (e.g., 'Western Australia'). If provided, rules specific to this state and visa subclass will be fetched."),
    # points_data_from_request_body: Optional[CalculatedPointsBodySchema] = Body(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    # Verify if the current user has access to this data
    if not verify_access(current_user, uploader_email):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this data"
        )
    
    # logger.info(f"Received request for /visa/analysis. Parsed calculated_points from body: {points_data_from_request_body.model_dump_json(indent=2)}")
    
    # N8N_WEBHOOK_URL = "https://zjpxpyanroshiskdre.app.n8n.cloud/webhook-test/228754dd-ab65-4559-8654-34c2f6f08f31"
    N8N_WEBHOOK_URL = "https://zjpxpyanroshiskdre.app.n8n.cloud/webhook/228754dd-ab65-4559-8654-34c2f6f08f31"
    logger.info(f"Request to get raw webhook response for candidate '{candidate_email}' (uploader: '{uploader_email}') via n8n.")

    interactions_collection = db['additional_info']

    # --- 1. Prepare the payload for the webhook (same logic as before) ---
    if resume_collection is None:
        logger.error("resume_collection is not initialized.")
        raise HTTPException(status_code=503, detail="Database service (resume_collection) unavailable.")
    if visa_collection is None:
        logger.error("visa_collection is not initialized.")
        raise HTTPException(status_code=503, detail="Database service (visa_collection) unavailable.")

    uploader_email_lower = uploader_email.lower()
    candidate_email_lower = candidate_email.lower()

    found_user_info_object: Optional[UserInfoFlexible] = None
    try:
        uploader_doc = resume_collection.find_one(
            {"_id": uploader_email_lower, "candidates.candidate_email": candidate_email_lower},
            {"candidates.$": 1}
        )
        try:
            qa_doc = interactions_collection.find_one(
                {"uploader_email": uploader_email_lower, "candidate_email": candidate_email_lower},
                {"additional_info.$": 1}
            )
        except:
            print('no quetions asked')
        if not uploader_doc or 'candidates' not in uploader_doc or not uploader_doc['candidates']:
            raise HTTPException(status_code=404, detail="Candidate resume data not found.")
        candidate_entry = uploader_doc['candidates'][0]
        raw_candidate_data = candidate_entry
        if qa_doc:
            candidate_answers = qa_doc['additional_info']
            raw_candidate_answers = candidate_answers
        else:
            raw_candidate_answers = {}
        # raw_additional_info = candidate_entry.get('additional_info')
        if not raw_candidate_data or not isinstance(raw_candidate_data, dict):
            raise HTTPException(status_code=404, detail="Candidate's parsed resume data is missing or not in expected dict format.")
        found_user_info_object = UserInfoFlexible(
            uploader_email=uploader_email, candidate_email=candidate_email, candidate_data=raw_candidate_data, questions_candidate_answered=raw_candidate_answers
        )
    except OperationFailure as op_err:
        logger.error(f"DB error fetching candidate data: {op_err.details}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"DB error fetching candidate data: {op_err.code_name}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error fetching candidate data: {e}")
        raise HTTPException(status_code=500, detail="Internal error fetching candidate data.")

    found_visa_rules_dict: Optional[Dict[str, Any]] = None
    try:
        visa_rules_query: Dict[str, Any] = {"visa_subclass": visa_subclass}
        if state: visa_rules_query["state"] = state
        visa_rule_doc_raw = visa_collection.find_one(visa_rules_query)
        if not visa_rule_doc_raw:
            detail_msg = f"Visa rules not found for subclass '{visa_subclass}'"
            if state: detail_msg += f" and state '{state}'"
            raise HTTPException(status_code=404, detail=detail_msg + ".")
        if "_id" in visa_rule_doc_raw and not isinstance(visa_rule_doc_raw["_id"], str):
            visa_rule_doc_raw["_id"] = str(visa_rule_doc_raw["_id"])
        found_visa_rules_dict = visa_rule_doc_raw
    except OperationFailure as op_err:
        logger.error(f"DB error fetching visa rules: {op_err.details}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"DB error fetching visa rules: {op_err.code_name}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error fetching visa rules: {e}")
        raise HTTPException(status_code=500, detail="Internal error fetching visa rules.")

    if found_user_info_object is None or found_visa_rules_dict is None:
        raise HTTPException(status_code=500, detail="Internal error: data fetching for webhook payload incomplete.")

    # if points_data_from_request_body is None:
    webhook_payload = SkillVisaEvaluationResponseFlexible(
        user_info=[found_user_info_object],
        visa_rules=found_visa_rules_dict,
        nominated_occupation=nominated_occupation,
        visa_subclass=visa_subclass
    )
    # else:
    #     webhook_payload = SkillVisaEvaluationResponseFlexible(
    #         user_info=[found_user_info_object],
    #         visa_rules=found_visa_rules_dict,
    #         nominated_occupation=nominated_occupation,
    #         visa_subclass=visa_subclass,
    #         calculated_points=points_data_from_request_body.model_dump()
    #     )
    # logger.info(f"Webhook payload successfully prepared for candidate '{candidate_email}'.")

    # --- 2. Call the n8n Webhook and return its response as-is ---
    try:
        logger.info(f"Sending data to n8n webhook: {N8N_WEBHOOK_URL}")
        n8n_response = requests.post(
            N8N_WEBHOOK_URL,
            json=webhook_payload.model_dump(exclude_none=True), # Send the dict representation
            timeout=50
        )
        n8n_response.raise_for_status()
        
        raw_webhook_data = n8n_response.json()
        
        logger.info(f"Successfully received and parsed response from n8n webhook. Status: {n8n_response.status_code}.")
        return raw_webhook_data

    except requests.exceptions.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from webhook: {e}. Response text: {n8n_response.text if 'webhook_response' in locals() else 'N/A'}")
        raise HTTPException(status_code=502, detail="Webhook service returned non-JSON response.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling n8n webhook: {e}")
        if e.response is not None:
            logger.error(f"Webhook response status: {e.response.status_code}, Text: {e.response.text}")
            if 400 <= e.response.status_code < 600:
                 raise HTTPException(status_code=502, detail=f"Webhook service error: {e.response.status_code} - {e.response.reason}")
        raise HTTPException(status_code=502, detail="Failed to communicate with webhook service.")
    except Exception as e_general: # Catch any other unexpected errors
        logger.exception(f"An unexpected error occurred during or after webhook call: {e_general}")
        raise HTTPException(status_code=500, detail="Internal server error processing webhook interaction.")

@app.get(
    "/api/evaluate/{uploader_email}/{candidate_email}/visa/analysis", # Modified path for clarity
    response_model=Any, # FastAPI endpoint will now return the raw JSON from webhook (parsed to Python object)
    tags=["Visa Evaluation"],
    summary="Triggers an n8n webhook and returns its raw JSON response",
    description="Combines candidate data with visa rules, sends it to an n8n webhook, and returns the entire JSON response received from the webhook as-is.",
    responses={
        200: {"description": "Successfully retrieved raw response from the webhook"},
        404: {"description": "Candidate, Uploader, or Visa Rule (for internal payload) not found"},
        500: {"description": "Internal server error"},
        502: {"description": "Bad Gateway - Error communicating with or non-JSON response from webhook service"},
        503: {"description": "Database service unavailable"}
    }
)
async def get_raw_webhook_response(
    uploader_email: str = Path(..., description="Email of the uploader."),
    candidate_email: str = Path(..., description="Email of the candidate."),
    nominated_occupation: str = Query(..., description="The occupation nominated by the candidate."),
    visa_subclass: str = Query(..., description="The visa subclass the candidate is applying for."),
    state: Optional[str] = Query(None, description="The state for state-specific visa rules (e.g., 'Western Australia'). If provided, rules specific to this state and visa subclass will be fetched."),
    # points_data_from_request_body: Optional[CalculatedPointsBodySchema] = Body(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    # Verify if the current user has access to this data
    if not verify_access(current_user, uploader_email):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this data"
        )
    
    # logger.info(f"Received request for /visa/analysis. Parsed calculated_points from body: {points_data_from_request_body.model_dump_json(indent=2)}")
    
    # N8N_WEBHOOK_URL = "https://zjpxpyanroshiskdre.app.n8n.cloud/webhook-test/228754dd-ab65-4559-8654-34c2f6f08f16"
    N8N_WEBHOOK_URL = "https://zjpxpyanroshiskdre.app.n8n.cloud/webhook/228754dd-ab65-4559-8654-34c2f6f08f16"
    logger.info(f"Request to get raw webhook response for candidate '{candidate_email}' (uploader: '{uploader_email}') via n8n.")

    interactions_collection = db['additional_info']

    # --- 1. Prepare the payload for the webhook (same logic as before) ---
    if resume_collection is None:
        logger.error("resume_collection is not initialized.")
        raise HTTPException(status_code=503, detail="Database service (resume_collection) unavailable.")
    if visa_collection is None:
        logger.error("visa_collection is not initialized.")
        raise HTTPException(status_code=503, detail="Database service (visa_collection) unavailable.")

    uploader_email_lower = uploader_email.lower()
    candidate_email_lower = candidate_email.lower()

    found_user_info_object: Optional[UserInfoFlexible] = None
    try:
        uploader_doc = resume_collection.find_one(
            {"_id": uploader_email_lower, "candidates.candidate_email": candidate_email_lower},
            {"candidates.$": 1}
        )
        try:
            qa_doc = interactions_collection.find_one(
                {"uploader_email": uploader_email_lower, "candidate_email": candidate_email_lower},
                {"additional_info.$": 1}
            )
        except:
            print('no quetions asked')
        if not uploader_doc or 'candidates' not in uploader_doc or not uploader_doc['candidates']:
            raise HTTPException(status_code=404, detail="Candidate resume data not found.")
        candidate_entry = uploader_doc['candidates'][0]
        raw_candidate_data = candidate_entry
        if qa_doc:
            candidate_answers = qa_doc['additional_info']
            raw_candidate_answers = candidate_answers
        else:
            raw_candidate_answers = {}
        # raw_additional_info = candidate_entry.get('additional_info')
        if not raw_candidate_data or not isinstance(raw_candidate_data, dict):
            raise HTTPException(status_code=404, detail="Candidate's parsed resume data is missing or not in expected dict format.")
        found_user_info_object = UserInfoFlexible(
            uploader_email=uploader_email, candidate_email=candidate_email, candidate_data=raw_candidate_data, questions_candidate_answered=raw_candidate_answers
        )
    except OperationFailure as op_err:
        logger.error(f"DB error fetching candidate data: {op_err.details}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"DB error fetching candidate data: {op_err.code_name}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error fetching candidate data: {e}")
        raise HTTPException(status_code=500, detail="Internal error fetching candidate data.")

    found_visa_rules_dict: Optional[Dict[str, Any]] = None
    try:
        visa_rules_query: Dict[str, Any] = {"visa_subclass": visa_subclass}
        if state: visa_rules_query["state"] = state
        visa_rule_doc_raw = visa_collection.find_one(visa_rules_query)
        if not visa_rule_doc_raw:
            detail_msg = f"Visa rules not found for subclass '{visa_subclass}'"
            if state: detail_msg += f" and state '{state}'"
            raise HTTPException(status_code=404, detail=detail_msg + ".")
        if "_id" in visa_rule_doc_raw and not isinstance(visa_rule_doc_raw["_id"], str):
            visa_rule_doc_raw["_id"] = str(visa_rule_doc_raw["_id"])
        found_visa_rules_dict = visa_rule_doc_raw
    except OperationFailure as op_err:
        logger.error(f"DB error fetching visa rules: {op_err.details}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"DB error fetching visa rules: {op_err.code_name}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error fetching visa rules: {e}")
        raise HTTPException(status_code=500, detail="Internal error fetching visa rules.")

    if found_user_info_object is None or found_visa_rules_dict is None:
        raise HTTPException(status_code=500, detail="Internal error: data fetching for webhook payload incomplete.")

    # if points_data_from_request_body is None:
    webhook_payload = SkillVisaEvaluationResponseFlexible(
        user_info=[found_user_info_object],
        visa_rules=found_visa_rules_dict,
        nominated_occupation=nominated_occupation,
        visa_subclass=visa_subclass
    )
    # else:
    #     webhook_payload = SkillVisaEvaluationResponseFlexible(
    #         user_info=[found_user_info_object],
    #         visa_rules=found_visa_rules_dict,
    #         nominated_occupation=nominated_occupation,
    #         visa_subclass=visa_subclass,
    #         calculated_points=points_data_from_request_body.model_dump()
    #     )
    # logger.info(f"Webhook payload successfully prepared for candidate '{candidate_email}'.")

    # --- 2. Call the n8n Webhook and return its response as-is ---
    try:
        logger.info(f"Sending data to n8n webhook: {N8N_WEBHOOK_URL}")
        n8n_response = requests.post(
            N8N_WEBHOOK_URL,
            json=webhook_payload.model_dump(exclude_none=True), # Send the dict representation
            timeout=50
        )
        n8n_response.raise_for_status()
        
        raw_webhook_data = n8n_response.json()
        
        logger.info(f"Successfully received and parsed response from n8n webhook. Status: {n8n_response.status_code}.")
        return raw_webhook_data

    except requests.exceptions.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from webhook: {e}. Response text: {n8n_response.text if 'webhook_response' in locals() else 'N/A'}")
        raise HTTPException(status_code=502, detail="Webhook service returned non-JSON response.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling n8n webhook: {e}")
        if e.response is not None:
            logger.error(f"Webhook response status: {e.response.status_code}, Text: {e.response.text}")
            if 400 <= e.response.status_code < 600:
                 raise HTTPException(status_code=502, detail=f"Webhook service error: {e.response.status_code} - {e.response.reason}")
        raise HTTPException(status_code=502, detail="Failed to communicate with webhook service.")
    except Exception as e_general: # Catch any other unexpected errors
        logger.exception(f"An unexpected error occurred during or after webhook call: {e_general}")
        raise HTTPException(status_code=500, detail="Internal server error processing webhook interaction.")

# @app.get(
#     "/api/evaluate/{uploader_email}/{candidate_email}/visa/points-calculation", # Modified path for clarity
#     response_model=Any, # FastAPI endpoint will now return the raw JSON from webhook (parsed to Python object)
#     tags=["Visa Evaluation"],
#     summary="Triggers an n8n webhook and returns its raw JSON response",
#     description="Combines candidate data with visa rules, sends it to an n8n webhook, and returns the entire JSON response received from the webhook as-is.",
#     responses={
#         200: {"description": "Successfully retrieved raw response from the webhook"},
#         404: {"description": "Candidate, Uploader, or Visa Rule (for internal payload) not found"},
#         500: {"description": "Internal server error"},
#         502: {"description": "Bad Gateway - Error communicating with or non-JSON response from webhook service"},
#         503: {"description": "Database service unavailable"}
#     }
# )
# async def get_raw_webhook_response( # Renamed function for clarity
#     uploader_email: str = Path(..., description="Email of the uploader."),
#     candidate_email: str = Path(..., description="Email of the candidate."),
#     nominated_occupation: str = Query(..., description="The occupation nominated by the candidate."),
#     visa_subclass: str = Query(..., description="The visa subclass the candidate is applying for."),
#     state: Optional[str] = Query(None, description="The state for state-specific visa rules (e.g., 'Western Australia'). If provided, rules specific to this state and visa subclass will be fetched."),
#     current_user: Dict[str, Any] = Depends(get_current_user)
# ):
#     # Verify if the current user has access to this data
#     if not verify_access(current_user, uploader_email):
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="You don't have permission to access this data"
#         )
    
#     # N8N_WEBHOOK_URL = "https://zjpxpyanroshiskdre.app.n8n.cloud/webhook-test/228754dd-ab65-4559-8654-34c2f6f08f17"
#     N8N_WEBHOOK_URL = "https://zjpxpyanroshiskdre.app.n8n.cloud/webhook/228754dd-ab65-4559-8654-34c2f6f08f17"
#     logger.info(f"Request to get raw webhook response for candidate '{candidate_email}' (uploader: '{uploader_email}') via n8n.")

#     interactions_collection = db['additional_info']

#     # --- 1. Prepare the payload for the webhook (same logic as before) ---
#     if resume_collection is None:
#         logger.error("resume_collection is not initialized.")
#         raise HTTPException(status_code=503, detail="Database service (resume_collection) unavailable.")
#     if visa_collection is None:
#         logger.error("visa_collection is not initialized.")
#         raise HTTPException(status_code=503, detail="Database service (visa_collection) unavailable.")

#     uploader_email_lower = uploader_email.lower()
#     candidate_email_lower = candidate_email.lower()

#     found_user_info_object: Optional[UserInfoFlexible] = None
#     try:
#         uploader_doc = resume_collection.find_one(
#             {"_id": uploader_email_lower, "candidates.candidate_email": candidate_email_lower},
#             {"candidates.$": 1}
#         )
#         try:
#             qa_doc = interactions_collection.find_one(
#                 {"uploader_email": uploader_email_lower, "candidate_email": candidate_email_lower},
#                 {"additional_info.$": 1}
#             )
#         except:
#             print('no quetions asked')
#         if not uploader_doc or 'candidates' not in uploader_doc or not uploader_doc['candidates']:
#             raise HTTPException(status_code=404, detail="Candidate resume data not found.")
#         candidate_entry = uploader_doc['candidates'][0]
#         raw_candidate_data = candidate_entry
#         if qa_doc:
#             candidate_answers = qa_doc['additional_info']
#             raw_candidate_answers = candidate_answers
#         else:
#             raw_candidate_answers = {}
#         # raw_additional_info = candidate_entry.get('additional_info')
#         if not raw_candidate_data or not isinstance(raw_candidate_data, dict):
#             raise HTTPException(status_code=404, detail="Candidate's parsed resume data is missing or not in expected dict format.")
#         found_user_info_object = UserInfoFlexible(
#             uploader_email=uploader_email, candidate_email=candidate_email, candidate_data=raw_candidate_data, questions_candidate_answered=raw_candidate_answers
#         )
#     except OperationFailure as op_err:
#         logger.error(f"DB error fetching candidate data: {op_err.details}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"DB error fetching candidate data: {op_err.code_name}")
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.exception(f"Unexpected error fetching candidate data: {e}")
#         raise HTTPException(status_code=500, detail="Internal error fetching candidate data.")

#     found_visa_rules_dict: Optional[Dict[str, Any]] = None
#     try:
#         visa_rules_query: Dict[str, Any] = {"visa_subclass": visa_subclass}
#         if state: visa_rules_query["state"] = state
#         visa_rule_doc_raw = visa_collection.find_one(visa_rules_query)
#         if not visa_rule_doc_raw:
#             detail_msg = f"Visa rules not found for subclass '{visa_subclass}'"
#             if state: detail_msg += f" and state '{state}'"
#             raise HTTPException(status_code=404, detail=detail_msg + ".")
#         if "_id" in visa_rule_doc_raw and not isinstance(visa_rule_doc_raw["_id"], str):
#             visa_rule_doc_raw["_id"] = str(visa_rule_doc_raw["_id"])
#         found_visa_rules_dict = visa_rule_doc_raw
#     except OperationFailure as op_err:
#         logger.error(f"DB error fetching visa rules: {op_err.details}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"DB error fetching visa rules: {op_err.code_name}")
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.exception(f"Unexpected error fetching visa rules: {e}")
#         raise HTTPException(status_code=500, detail="Internal error fetching visa rules.")

#     if found_user_info_object is None or found_visa_rules_dict is None:
#         raise HTTPException(status_code=500, detail="Internal error: data fetching for webhook payload incomplete.")

#     webhook_payload = SkillVisaEvaluationResponseFlexible(
#         user_info=[found_user_info_object],
#         visa_rules=found_visa_rules_dict,
#         nominated_occupation=nominated_occupation,
#         visa_subclass=visa_subclass
#     )
#     logger.info(f"Webhook payload successfully prepared for candidate '{candidate_email}'.")

#     # --- 2. Call the n8n Webhook and return its response as-is ---
#     try:
#         logger.info(f"Sending data to n8n webhook: {N8N_WEBHOOK_URL} and awaiting its raw response.")
#         webhook_response = requests.post(N8N_WEBHOOK_URL, json=webhook_payload.dict(), timeout=50)
#         webhook_response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        
#         # Parse the JSON response from the webhook
#         raw_webhook_data = webhook_response.json() 
        
#         logger.info(f"Successfully received and parsed response from n8n webhook. Status: {webhook_response.status_code}.")
#         return raw_webhook_data # Return the parsed JSON object directly

#     except requests.exceptions.JSONDecodeError as e:
#         logger.error(f"Failed to decode JSON response from webhook: {e}. Response text: {webhook_response.text if 'webhook_response' in locals() else 'N/A'}")
#         raise HTTPException(status_code=502, detail="Webhook service returned non-JSON response.")
#     except requests.exceptions.RequestException as e:
#         logger.error(f"Error calling n8n webhook: {e}")
#         if e.response is not None:
#             logger.error(f"Webhook response status: {e.response.status_code}, Text: {e.response.text}")
#             if 400 <= e.response.status_code < 600:
#                  raise HTTPException(status_code=502, detail=f"Webhook service error: {e.response.status_code} - {e.response.reason}")
#         raise HTTPException(status_code=502, detail="Failed to communicate with webhook service.")
#     except Exception as e_general: # Catch any other unexpected errors
#         logger.exception(f"An unexpected error occurred during or after webhook call: {e_general}")
#         raise HTTPException(status_code=500, detail="Internal server error processing webhook interaction.")

@app.get(
    "/api/evaluate/{uploader_email}/{candidate_email}/visa/points-calculation",
    response_model=Any,
    tags=["Visa Evaluation"],
    summary="Triggers an n8n webhook, returns its raw JSON response, and upserts it into MongoDB.",
    description="Combines candidate data with visa rules, sends it to an n8n webhook, returns the entire JSON response. This response is then stored in the 'points_calculation' collection, overwriting any previous calculation for the same uploader, candidate, occupation, visa subclass, and state.",
    responses={
        200: {"description": "Successfully retrieved raw response from the webhook (and upserted it)"},
        404: {"description": "Candidate, Uploader, or Visa Rule (for internal payload) not found"},
        500: {"description": "Internal server error (could include DB write failure if critical)"},
        502: {"description": "Bad Gateway - Error communicating with or non-JSON response from webhook service"},
        503: {"description": "Database service unavailable"}
    }
)
async def get_raw_webhook_response_and_upsert( # Renamed for clarity
    uploader_email: str = Path(..., description="Email of the uploader."),
    candidate_email: str = Path(..., description="Email of the candidate."),
    nominated_occupation: str = Query(..., description="The occupation nominated by the candidate."),
    visa_subclass: str = Query(..., description="The visa subclass the candidate is applying for."),
    state: Optional[str] = Query(None, description="The state for state-specific visa rules (e.g., 'Western Australia'). If provided, rules specific to this state and visa subclass will be fetched."),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    # Verify if the current user has access to this data
    if not verify_access(current_user, uploader_email):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this data"
        )

    # N8N_WEBHOOK_URL = "https://zjpxpyanroshiskdre.app.n8n.cloud/webhook-test/228754dd-ab65-4559-8654-34c2f6f08f18"
    # N8N_WEBHOOK_URL = "https://zjpxpyanroshiskdre.app.n8n.cloud/webhook/228754dd-ab65-4559-8654-34c2f6f08f18"
    # N8N_WEBHOOK_URL = "https://zjpxpyanroshiskdre.app.n8n.cloud/webhook-test/228754dd-ab65-4559-8654-34c2f6f08f19"
    N8N_WEBHOOK_URL = "https://zjpxpyanroshiskdre.app.n8n.cloud/webhook/228754dd-ab65-4559-8654-34c2f6f08f19"
    logger.info(f"Request for points calculation for candidate '{candidate_email}' (uploader: '{uploader_email}') via n8n.")

    # --- Get MongoDB collections ---
    if db is None:
        logger.error("Database connection (db) is not initialized.")
        raise HTTPException(status_code=503, detail="Database service unavailable.")

    interactions_collection = db['additional_info']
    points_calculation_collection = db['points_calculation']

    # --- 1. Prepare the payload for the webhook (same logic as before) ---
    if resume_collection is None:
        logger.error("resume_collection is not initialized.")
        raise HTTPException(status_code=503, detail="Database service (resume_collection) unavailable.")
    if visa_collection is None:
        logger.error("visa_collection is not initialized.")
        raise HTTPException(status_code=503, detail="Database service (visa_collection) unavailable.")

    uploader_email_lower = uploader_email.lower()
    candidate_email_lower = candidate_email.lower()

    found_user_info_object: Optional[UserInfoFlexible] = None
    try:
        uploader_doc = resume_collection.find_one(
            {"_id": uploader_email_lower, "candidates.candidate_email": candidate_email_lower},
            {"candidates.$": 1}
        )
        qa_doc = None
        try:
            qa_doc = interactions_collection.find_one(
                {"uploader_email": uploader_email_lower, "candidate_email": candidate_email_lower},
                {"additional_info.$": 1}
            )
        except Exception as e_qa:
            logger.warning(f"Could not fetch QA doc for {candidate_email_lower} from {uploader_email_lower}: {e_qa}")

        if not uploader_doc or 'candidates' not in uploader_doc or not uploader_doc['candidates']:
            raise HTTPException(status_code=404, detail="Candidate resume data not found.")
        
        candidate_entry = uploader_doc['candidates'][0]
        raw_candidate_data = candidate_entry
        raw_candidate_answers = {}
        if qa_doc and 'additional_info' in qa_doc and qa_doc['additional_info']:
            candidate_answers_data = qa_doc['additional_info']
            if isinstance(candidate_answers_data, list) and candidate_answers_data:
                 raw_candidate_answers = candidate_answers_data[0]
            elif isinstance(candidate_answers_data, dict): # Should not happen with .$ projection but good to have a fallback
                 raw_candidate_answers = candidate_answers_data
            else:
                logger.warning(f"Unexpected structure for additional_info in qa_doc for {candidate_email_lower}. Data: {candidate_answers_data}")

        if not raw_candidate_data or not isinstance(raw_candidate_data, dict):
            raise HTTPException(status_code=404, detail="Candidate's parsed resume data is missing or not in expected dict format.")
        
        found_user_info_object = UserInfoFlexible(
            uploader_email=uploader_email,
            candidate_email=candidate_email,
            candidate_data=raw_candidate_data,
            questions_candidate_answered=raw_candidate_answers
        )
    except OperationFailure as op_err:
        logger.error(f"DB error fetching candidate data: {op_err.details}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"DB error fetching candidate data: {op_err.code_name}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error fetching candidate data: {e}")
        raise HTTPException(status_code=500, detail="Internal error fetching candidate data.")

    found_visa_rules_dict: Optional[Dict[str, Any]] = None
    try:
        visa_rules_query: Dict[str, Any] = {"visa_subclass": "189"}
        if state: visa_rules_query["state"] = state
        visa_rule_doc_raw = visa_collection.find_one(visa_rules_query)
        if not visa_rule_doc_raw:
            detail_msg = f"Visa rules not found for subclass '{visa_subclass}'"
            if state: detail_msg += f" and state '{state}'"
            raise HTTPException(status_code=404, detail=detail_msg + ".")
        if "_id" in visa_rule_doc_raw and not isinstance(visa_rule_doc_raw["_id"], str):
            visa_rule_doc_raw["_id"] = str(visa_rule_doc_raw["_id"])
        found_visa_rules_dict = visa_rule_doc_raw
    except OperationFailure as op_err:
        logger.error(f"DB error fetching visa rules: {op_err.details}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"DB error fetching visa rules: {op_err.code_name}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error fetching visa rules: {e}")
        raise HTTPException(status_code=500, detail="Internal error fetching visa rules.")

    if found_user_info_object is None or found_visa_rules_dict is None:
        logger.error("Data fetching for webhook payload incomplete, though no specific error was raised.")
        raise HTTPException(status_code=500, detail="Internal error: data fetching for webhook payload incomplete.")

    webhook_payload = SkillVisaEvaluationResponseFlexible(
        user_info=[found_user_info_object],
        visa_rules=found_visa_rules_dict,
        nominated_occupation=nominated_occupation,
        visa_subclass=visa_subclass
    )
    logger.info(f"Webhook payload successfully prepared for candidate '{candidate_email}'.")

    # --- 2. Call the n8n Webhook ---
    try:
        logger.info(f"Sending data to n8n webhook: {N8N_WEBHOOK_URL} and awaiting its raw response.")
        n8n_response = requests.post(N8N_WEBHOOK_URL, json=webhook_payload.model_dump(exclude_none=True), timeout=50)
        n8n_response.raise_for_status()
        
        raw_webhook_data = n8n_response.json()
        
        logger.info(f"Successfully received and parsed response from n8n webhook. Status: {n8n_response.status_code}.")

        # --- 3. Upsert the raw webhook response in MongoDB ---
        try:
            # Define the filter to find the document to update/replace
            query_filter = {
                "uploader_email": uploader_email_lower,
                "candidate_email": candidate_email_lower,
                "nominated_occupation": nominated_occupation,
                "visa_subclass": visa_subclass,
                "state": state if state else None, # Match state or absence of state
            }

            # Define the replacement document (the entire new document)
            replacement_document = {
                "uploader_email": uploader_email_lower,
                "candidate_email": candidate_email_lower,
                "nominated_occupation": nominated_occupation,
                "visa_subclass": visa_subclass,
                "state": state if state else None,
                "webhook_response_data": raw_webhook_data,
                "calculated_at": datetime.now(timezone.utc),
                "api_version": "v1_points_calculation"
            }

            upsert_result = points_calculation_collection.replace_one(
                query_filter,
                replacement_document,
                upsert=True
            )

            if upsert_result.upserted_id:
                logger.info(f"Successfully inserted new points calculation response in MongoDB with id: {upsert_result.upserted_id}")
            elif upsert_result.modified_count > 0:
                logger.info(f"Successfully updated existing points calculation response in MongoDB. Matched: {upsert_result.matched_count}, Modified: {upsert_result.modified_count}")
            else:
                logger.info(f"Points calculation response already up-to-date in MongoDB. Matched: {upsert_result.matched_count}")

        except OperationFailure as db_op_err:
            logger.error(f"MongoDB OperationFailure during upsert of points calculation: {db_op_err.details}", exc_info=True)
            # Decide if this is critical. For now, we'll just log and still return the data.
        except Exception as e_db:
            logger.error(f"Unexpected error upserting points calculation in MongoDB: {e_db}", exc_info=True)
            # Same as above, log and continue for now.

        return raw_webhook_data

    except requests.exceptions.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from webhook: {e}. Response text: {n8n_response.text if 'n8n_response' in locals() else 'N/A'}")
        raise HTTPException(status_code=502, detail="Webhook service returned non-JSON response.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling n8n webhook: {e}")
        if e.response is not None:
            logger.error(f"Webhook response status: {e.response.status_code}, Text: {e.response.text}")
            if 400 <= e.response.status_code < 600:
                 raise HTTPException(status_code=502, detail=f"Webhook service error: {e.response.status_code} - {e.response.reason}")
        raise HTTPException(status_code=502, detail="Failed to communicate with webhook service.")
    except Exception as e_general:
        logger.exception(f"An unexpected error occurred during or after webhook call: {e_general}")
        raise HTTPException(status_code=500, detail="Internal server error processing webhook interaction.")

class DetailEvaluationItem(BaseModel):
    criteria: str
    short_explanation: str
    score: int # Assuming score is always an integer

class PointsCalculationOutputSchema(BaseModel):
    detail_evaluation: List[DetailEvaluationItem] = Field(..., description="Detailed breakdown of points for each criteria.")
    total_score: int # Or float, adjust if n8n can return float
    overall_analysis: str

class CandidateStoredCalculation(BaseModel):
    output: PointsCalculationOutputSchema
    # You might want to include other identifying info from the stored document
    nominated_occupation: Optional[str] = None
    visa_subclass: Optional[str] = None
    state: Optional[str] = None
    calculated_at: Optional[datetime] = None


# ... (previous code for imports, Pydantic models, app setup) ...
@app.get(
    "/api/evaluate/{uploader_email}/{candidate_email}/visa/retrieve-all-points",
    response_model=List[CandidateStoredCalculation],
    tags=["Visa Evaluation"],
    summary="Retrieves all stored points calculations for a specific candidate.",
    description="Fetches all records from the 'points_calculation' collection for the given uploader and candidate, formatted as per the specified output structure.",
    responses={
        200: {"description": "Successfully retrieved stored points calculations."},
        403: {"description": "Forbidden - User does not have access."},
        404: {"description": "No points calculations found for this candidate."},
        500: {"description": "Internal server error."},
        503: {"description": "Database service unavailable."}
    }
)
async def get_candidate_points_calculations(
    uploader_email: str = Path(..., description="Email of the uploader."),
    candidate_email: str = Path(..., description="Email of the candidate."),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    if not verify_access(current_user, uploader_email):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this data."
        )

    if db is None:
        logger.error("Database connection (db) is not initialized for points retrieval.")
        raise HTTPException(status_code=503, detail="Database service unavailable.")
    
    points_calculation_collection = db['points_calculation']
    uploader_email_lower = uploader_email.lower()
    candidate_email_lower = candidate_email.lower()

    try:
        logger.info(f"Fetching stored points calculations for uploader '{uploader_email_lower}', candidate '{candidate_email_lower}'.")
        
        cursor = points_calculation_collection.find({
            "uploader_email": uploader_email_lower,
            "candidate_email": candidate_email_lower
        })

        results = []
        for doc in cursor:
            webhook_data_list = doc.get("webhook_response_data") # This is a list

            # **** MODIFICATION START ****
            if webhook_data_list and isinstance(webhook_data_list, list) and len(webhook_data_list) > 0:
                # Assuming the actual data is in the first element of the list
                # and that element has the 'output' key
                actual_webhook_content = webhook_data_list[0] 
                
                if actual_webhook_content and "output" in actual_webhook_content:
                    output_payload = actual_webhook_content.get("output")
                    if output_payload:
                        try:
                            # Parse the content of the 'output' key
                            output_data = PointsCalculationOutputSchema(**output_payload)
                            
                            item = CandidateStoredCalculation(
                                output=output_data,
                                nominated_occupation=doc.get("nominated_occupation"),
                                visa_subclass=doc.get("visa_subclass"),
                                state=doc.get("state"),
                                calculated_at=doc.get("calculated_at")
                            )
                            results.append(item)
                        except Exception as e_parse:
                            logger.error(f"Error parsing 'output' content for doc_id {doc.get('_id')}: {e_parse}. Data: {output_payload}")
                            continue
                    else:
                        logger.warning(f"'output' key is missing or empty in webhook_response_data[0] for doc_id {doc.get('_id')}")
                else:
                    logger.warning(f"First element of 'webhook_response_data' is missing 'output' key or is empty for doc_id {doc.get('_id')}. Data: {actual_webhook_content}")
            # **** MODIFICATION END ****
            elif webhook_data_list: # It exists but is not a non-empty list
                 logger.warning(f"Document {doc.get('_id')} for candidate {candidate_email_lower} has 'webhook_response_data' but it's not a non-empty list. Data: {webhook_data_list}")
            else:
                logger.warning(f"Document {doc.get('_id')} for candidate {candidate_email_lower} is missing 'webhook_response_data'.")


        if not results:
            logger.info(f"No successfully parsed points calculations found for uploader '{uploader_email_lower}', candidate '{candidate_email_lower}'. Returning empty list.")
            # If you want 404 when no valid data is parsed:
            # raise HTTPException(status_code=404, detail="No valid points calculations found for this candidate.")

        return results

    except OperationFailure as db_op_err:
        logger.error(f"MongoDB OperationFailure fetching points calculations: {db_op_err.details}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error while fetching calculations.")
    except Exception as e:
        logger.exception(f"Unexpected error fetching points calculations for candidate '{candidate_email_lower}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


class AnswerItem(BaseModel):
    question: str
    answer: str

class CandidateResponse(BaseModel):
    answers: List[AnswerItem]

class AdditionalInfoData(BaseModel):
    candidate_responses: List[CandidateResponse]

class DocumentModel(BaseModel):
    id: str = Field(..., alias="_id") # Use alias for MongoDB's _id
    candidate_email: EmailStr
    uploader_email: EmailStr
    additional_info: Optional[AdditionalInfoData] = None # Make optional if might be missing
    created_at: datetime
    last_updated_at: datetime

    class Config:
        populate_by_name = True # Allow using '_id' during initialization
        json_encoders = {ObjectId: str, datetime: lambda dt: dt.isoformat()} # Serialize ObjectId and datetime
        # If using FastAPI < 0.100.0, use 'allow_population_by_field_name' instead of 'populate_by_name'

class UpdateAnswerRequest(BaseModel):
    question: str
    answer: str

class BulkUpdateAnswerRequest(BaseModel):
    updates: List[UpdateAnswerRequest]

@app.get("/user-answers",
         response_model=List[AnswerItem],
         summary="Get latest answers for a specific uploader-candidate pair")
async def get_user_answers(
    uploader_email: EmailStr = Query(..., description="Email of the uploader"),
    candidate_email: EmailStr = Query(..., description="Email of the candidate"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    # Verify if the current user has access to this data
    if not verify_access(current_user, uploader_email):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this data"
        )
    """
    Retrieves the latest question-answer pairs for a specific interaction
    identified by uploader and candidate emails, handling different historical
    schema formats for 'candidate_responses'.
    """
    uploader_email_lower = str(uploader_email).lower()
    candidate_email_lower = str(candidate_email).lower()
    query = {"uploader_email": uploader_email_lower, "candidate_email": candidate_email_lower}
    interactions_collection = db['additional_info']

    logger.info(f"Fetching answers for query: {query}")

    try:
        # Use await with motor find_one
        doc = interactions_collection.find_one(query)

        if not doc:
            logger.warning(f"No interaction document found for query: {query}")
            # Return empty list if no interaction means no answers
            return []

        logger.info(f"Document found for {candidate_email_lower}. Processing additional_info...")
        latest_answers: Dict[str, str] = {}
        additional_info = doc.get("additional_info")

        if not additional_info or not isinstance(additional_info, dict):
            logger.warning(f"No 'additional_info' dictionary found or it's not a dict for query: {query}. Doc ID: {doc.get('_id')}")
            return [] # No info means no answers

        candidate_responses_data = additional_info.get("candidate_responses")
        data_type = type(candidate_responses_data).__name__
        logger.info(f"Found 'candidate_responses' field with type: {data_type}")

        # --- Helper function to process a single 'answers' list ---
        def process_answers_list(answers_list: List[Any]):
            if not isinstance(answers_list, list):
                logger.warning(f"Expected a list for 'answers', but got {type(answers_list)}. Skipping.")
                return

            processed_count = 0
            for index, answer_item in enumerate(answers_list):
                if isinstance(answer_item, dict):
                    question = answer_item.get("question")
                    answer = answer_item.get("answer") # Get answer, could be None

                    if isinstance(question, str) and question.strip(): # Ensure question is a non-empty string
                        # Store the answer, converting None to empty string for consistency
                        latest_answers[question.strip()] = answer if answer is not None else ""
                        processed_count += 1
                    else:
                         logger.warning(f"Invalid or missing 'question' at index {index} in answers list. Item: {answer_item}")
                else:
                     logger.warning(f"Item at index {index} in answers list is not a dictionary. Item: {answer_item}")
            logger.info(f"Processed {processed_count} valid Q&A pairs from an answers list.")


        # --- Check the type of candidate_responses_data ---
        if isinstance(candidate_responses_data, list):
            logger.info("Processing 'candidate_responses' as a LIST.")
            # Iterate through response sets in order (older to newer)
            for i, response_set in enumerate(candidate_responses_data):
                if isinstance(response_set, dict):
                    answers = response_set.get("answers")
                    if answers:
                        logger.debug(f"Processing answers list from response_set index {i}")
                        process_answers_list(answers)
                    else:
                        logger.warning(f"Missing 'answers' key in response_set at index {i}. Set: {response_set}")
                else:
                     logger.warning(f"Item in 'candidate_responses' list at index {i} is not a dictionary. Item: {response_set}")

        elif isinstance(candidate_responses_data, dict):
            logger.info("Processing 'candidate_responses' as a DICTIONARY (older format).")
            # Directly look for the 'answers' key within this dictionary
            answers = candidate_responses_data.get("answers")
            if answers:
                 logger.debug("Processing answers list from the dictionary format.")
                 process_answers_list(answers)
            else:
                logger.warning(f"Missing 'answers' key in 'candidate_responses' dictionary. Dict: {candidate_responses_data}")

        else:
            # Handle None or other unexpected types
            logger.warning(f"'candidate_responses' field is None or an unexpected type: {data_type}. Cannot process answers.")
            # Depending on requirements, you might return [] here or continue if other fields could contain answers

        logger.info(f"Finished processing. Found {len(latest_answers)} unique questions.")

        # Convert the dictionary of latest answers back to a list of AnswerItem objects
        result_list = [AnswerItem(question=q, answer=a) for q, a in latest_answers.items()]

        # Optional: Sort results alphabetically by question
        result_list.sort(key=lambda item: item.question)

        logger.info(f"Returning {len(result_list)} AnswerItems.")
        return result_list

    except Exception as e:
        # Log the full error and raise a generic 500
        logger.exception(f"An unexpected error occurred while fetching answers for query {query}: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# 2. Update/Add Answers for a User Pair
@app.put("/user-answers",
         summary="Update or add answers for a specific uploader-candidate pair",
         status_code=200)
async def update_user_answers(
    updates: BulkUpdateAnswerRequest = Body(...),
    uploader_email: EmailStr = Query(..., description="Email of the uploader identifying the document"),
    candidate_email: EmailStr = Query(..., description="Email of the candidate identifying the document"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    # Verify if the current user has access to this data
    if not verify_access(current_user, uploader_email):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this data"
        )
    """
    Updates answers for existing questions or adds new question-answer pairs
    to the *last* entry in the `candidate_responses` array for the document
    matching the provided uploader and candidate emails.

    - Finds the document based on `uploader_email` and `candidate_email`.
    - If a question in the request already exists in the *last* response set, its answer is updated.
    - If a question does not exist in the *last* response set, it's added as a new entry
      in the `answers` list of that last `candidate_responses` item.
    - If the document doesn't exist, returns 404.
    - If `additional_info` or `candidate_responses` don't exist, they will be created.
    - Updates the `last_updated_at` timestamp.
    """
    interactions_collection = db['additional_info']
    query = {"uploader_email": uploader_email, "candidate_email": candidate_email}

    # --- Find the document ---
    doc = interactions_collection.find_one(query)
    if not doc:
        raise HTTPException(status_code=404, detail=f"No interaction found for uploader '{uploader_email}' and candidate '{candidate_email}' to update.")

    doc_id = doc["_id"] # Get the ObjectId for the update operation

    # --- Prepare for Update ---
    # Ensure structures exist, default to empty lists if necessary
    if "additional_info" not in doc or not isinstance(doc.get("additional_info"), dict):
        doc["additional_info"] = {"candidate_responses": []}

    if "candidate_responses" not in doc["additional_info"] or not isinstance(doc["additional_info"].get("candidate_responses"), list):
        doc["additional_info"]["candidate_responses"] = []

    # If there are no response sets yet, add one
    if not doc["additional_info"]["candidate_responses"]:
        doc["additional_info"]["candidate_responses"].append({"answers": []})

    # Target the *last* response set for modifications
    last_response_index = len(doc["additional_info"]["candidate_responses"]) - 1
    # Ensure the last item is a dict and has an 'answers' list
    if not isinstance(doc["additional_info"]["candidate_responses"][last_response_index], dict):
         doc["additional_info"]["candidate_responses"][last_response_index] = {"answers": []} # Replace if not dict
    target_answers = doc["additional_info"]["candidate_responses"][last_response_index].setdefault("answers", [])
    if not isinstance(target_answers, list): # Ensure 'answers' is a list
        target_answers = []
        doc["additional_info"]["candidate_responses"][last_response_index]["answers"] = target_answers


    # Create a lookup for faster checks within the target answers list
    existing_questions_map: Dict[str, Any] = {}
    # Need to handle potential non-dict items or items missing 'question' defensively
    for i, item in enumerate(target_answers):
        if isinstance(item, dict) and "question" in item:
            existing_questions_map[item["question"]] = item
        else:
            # Optionally log a warning or decide how to handle malformed data
            print(f"Warning: Malformed item found at index {i} in target answers for doc {doc_id}")


    updated = False
    for update_req in updates.updates:
        if update_req.question in existing_questions_map:
            # Update existing answer only if it actually changed
            if existing_questions_map[update_req.question].get("answer") != update_req.answer:
                existing_questions_map[update_req.question]["answer"] = update_req.answer
                updated = True
                print(f"Updated question: '{update_req.question}' in doc {doc_id}")
        else:
            # Add new question-answer pair
            new_qa = {"question": update_req.question, "answer": update_req.answer}
            target_answers.append(new_qa)
            # Also add to map for potential subsequent updates in the same request
            existing_questions_map[update_req.question] = new_qa
            updated = True
            print(f"Added new question: '{update_req.question}' to doc {doc_id}")

    # --- Perform Update if changes were made ---
    # if updated:
    current_time = datetime.utcnow()
    result = interactions_collection.update_one(
        {"_id": doc_id}, # Use the specific _id found earlier
        {"$set": {
            # Set the entire modified candidate_responses list
            "additional_info.candidate_responses": doc["additional_info"]["candidate_responses"],
            "last_updated_at": current_time
        }}
    )

    if result.modified_count >= 1: # Should be 1, but >= accounts for potential edge cases
            print(f"Successfully updated answers for doc {doc_id}")
            # Optionally return the updated data or just success
            # Fetching again might be cleaner if returning data:
            # updated_doc = await coll.find_one({"_id": doc_id})
            # return parse_obj_as(List[AnswerItem], updated_doc...) # adapt parsing
            return {"message": "Answers updated successfully."}
    else:
        # This might happen if the document was deleted between find_one and update_one
        # or if no actual changes were needed (though 'updated' flag should prevent this)
        print(f"Warning: Update operation reported no modifications for doc {doc_id}")
        # Decide if this is a client error (4xx) or server error (5xx)
        # If 'updated' was True, it suggests a server-side issue/race condition.
        raise HTTPException(status_code=500, detail="Failed to update document, inconsistent state detected or no changes applied.")
    # else:
    #     print(f"No changes needed for doc {doc_id}")
    #     return {"message": "No changes needed, answers are already up-to-date."}


@app.post(
    "/api/candidates/empty",
    summary="Create empty candidate data if it does not exist",
    status_code=200, # Use 200 for processed, message indicates if created/exists
    tags=["Candidate Data"],
    responses={
        200: {"description": "Operation successful", "content": {"application/json": {"example": {"status": "created", "message": "Empty data created...", "uploader_email": "uploader@example.com", "candidate_email": "candidate@example.com"}}}},
        400: {"description": "Invalid input data"},
        503: {"description": "Database service unavailable"},
        500: {"description": "Internal server error"}
    }
)
async def create_empty_candidate_data(
    request_body: Dict[str, str] = Body(..., description="Requires 'uploader_email' and 'candidate_email'"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    # Verify if the current user has access to this data
    if not verify_access(current_user, uploader_email):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this data"
        )
    """
    Creates an empty data structure for a specified uploader and candidate email
    in the 'uploaders_resumes' collection, but only if data for that specific
    uploader-candidate pair does not already exist.
    """
    if resume_collection is None:
        logger.error("Resume collection object is None. Cannot create empty data.")
        raise HTTPException(status_code=503, detail="Database service unavailable.")

    try:
        uploader_email = request_body.get("uploader_email")
        candidate_email = request_body.get("candidate_email")

        if not uploader_email or not candidate_email:
             raise HTTPException(status_code=400, detail="Both 'uploader_email' and 'candidate_email' are required in the request body.")

        # Basic email format validation (can use EmailStr from Pydantic Body but user requested no models)
        if "@" not in uploader_email or "@" not in candidate_email:
            raise HTTPException(status_code=400, detail="Invalid email format for uploader_email or candidate_email.")


        uploader_email_lower = uploader_email.lower()
        candidate_email_lower = candidate_email.lower()

        logger.info(f"Request to create empty data for candidate '{candidate_email_lower}' under uploader '{uploader_email_lower}'.")

        # 1. Check if the candidate entry already exists
        existing_doc = resume_collection.find_one(
            {"_id": uploader_email_lower, "candidates.candidate_email": candidate_email_lower},
            {"_id": 1} # Projection to make the check faster
        )

        if existing_doc:
            logger.info(f"Data already exists for candidate '{candidate_email_lower}' under uploader '{uploader_email_lower}'. Skipping creation.")
            return JSONResponse(
                status_code=200,
                content={
                    "status": "already_exists",
                    "message": f"Data already exists for candidate '{candidate_email}'. No changes made.",
                    "uploader_email": uploader_email,
                    "candidate_email": candidate_email
                }
            )

        # 2. If data doesn't exist, create the empty structure
        empty_data_structure = {
            "email": candidate_email_lower, # Fill only the email field
            "personal_details": {
              "name": "", "phone": "", "DOB": "", "gender": "",
              "maritalStatus": "", "nationality": "", "country_residency": "",
              "passportDetails": {"number": "", "expiryDate": "", "issuingCountry": ""}
            },
            "education": {
              "hasStudiedInAustralia": "", "numberOfQualificationsCompletedInAustralia": "",
              "questions_if_has_studied_in_australia": [],
              "australian_education": [], "overseas_education": []
            },
            "workExperience": {
              "australian_workExperience": [], "overseas_workExperience": []
            },
            "englishProficiency": {
              "englishLanguageTestCompleted": "",
              "englishExamDetails": {"examName": "", "examDate": "", "overallScore": "", "listeningScore": "", "readingScore": "", "speakingScore": "", "writingScore": ""},
              "estimatedProficiency": ""
            },
            "if_partner": {
              "workexperiance_in_last_5_years": { # Note: Kept your original key name with typo
                "australian_workExperience": [], # Using empty lists as default based on your example
                "overseas_workExperience": []
              },
              "estimatedProficiency": "" # Default to empty string based on example
            },
            "community_language_accreditation": {
              "holdsNAATIcertification": "", "canGiveCommunityLanguageClasses": ""
            },
            "living_in_australia": {
              "hasLivedInAustralia": "", # Default to empty string
              "numeberOfDiffrentStatesLivedIn": "", # Note: Kept original key name with typo
              "satesLivedIn": [] # Using empty list as default
            }
        }

        candidate_entry = {
            "candidate_email": candidate_email_lower,
            "parsed_resume_data": empty_data_structure
        }

        # 3. Insert the new candidate entry (upserting the uploader document if it doesn't exist)
        result = resume_collection.update_one(
            {"_id": uploader_email_lower},
            {
                "$push": {"candidates": candidate_entry},
                "$setOnInsert": {
                    "uploader_email": uploader_email_lower,
                     # Add a creation timestamp if desired, though _id already has one
                     # "created_at": datetime.utcnow()
                    }
            },
            upsert=True # Create uploader doc if it doesn't exist
        )

        if result.upserted_id:
            # A new uploader document was created and the candidate added
            logger.info(f"Created new uploader document '{uploader_email_lower}' and added empty data for candidate '{candidate_email_lower}'.")
            return JSONResponse(
                status_code=201, # Indicate resource created
                content={
                    "status": "created",
                    "message": f"New uploader record created. Empty data created for candidate '{candidate_email}'.",
                    "uploader_email": uploader_email,
                    "candidate_email": candidate_email
                }
            )
        elif result.modified_count > 0:
            # The uploader document existed, and the new candidate entry was added
            logger.info(f"Uploader document '{uploader_email_lower}' found. Added empty data for new candidate '{candidate_email_lower}'.")
            return JSONResponse(
                status_code=200, # Indicate success, modification happened
                content={
                    "status": "created", # Still report "created" from a data perspective
                    "message": f"Empty data created for candidate '{candidate_email}'.",
                    "uploader_email": uploader_email,
                    "candidate_email": candidate_email
                }
            )
        else:
            # This case is unlikely if the find_one didn't find it, but possible in race conditions
            # or if the upsert didn't actually modify anything (e.g., failed silently?)
            logger.error(f"Failed to create empty data for candidate '{candidate_email_lower}' under uploader '{uploader_email_lower}'. Upsert result: {result.raw_result}")
            raise HTTPException(status_code=500, detail="Failed to create empty data due to a database issue.")

    except OperationFailure as op_err:
        logger.error(f"MongoDB operation failed creating empty data for uploader '{uploader_email_lower}', candidate '{candidate_email_lower}': {op_err.details}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database operation error: {op_err.code_name}")
    except HTTPException:
        # Re-raise HTTPExceptions raised within the try block
        raise
    except Exception as e:
        logger.exception(f"Unexpected error creating empty data for uploader '{uploader_email_lower}', candidate '{candidate_email_lower}': {e}")
        raise HTTPException(status_code=500, detail="An unexpected internal server error occurred.")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Starting Uvicorn server on http://{host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True) # Ensure "main:app" matches your filename