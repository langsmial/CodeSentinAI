import json
import logging
import re
import uuid
from mitmproxy import http
from sentence_transformers import SentenceTransformer, util

class CodeSentinAI:
    def __init__(self):
        logging.info("===== CodeSentinAI: 최상위 하이브리드 엔진 로드 중 =====")
        
        # 1. 파인튜닝 완료된 로컬 SBERT 모델 로드
        self.model = SentenceTransformer('./fine_tuned_ko_sbert') #파인튜닝한 모델
        #self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') - 테스트용 다국어 모델
        #self.model = SentenceTransformer('jhgan/ko-sroberta-multitask') - 파인튜닝한 모델의 베이스 모델
        
        # 2. 30가지 사내 기밀 기준 문장
        self.sensitive_references = [
            "클라우드 서버 인프라 아키텍처 및 내부망 IP 주소 대역 정보",
            "사내 VPN 접속 정보 및 방화벽 우회 관련 네트워크 설정 데이터",
            "온프레미스(On-Premise) 서버 운영 환경 및 라우팅 테이블 구성도",
            "도커(Docker) 및 쿠버네티스(Kubernetes) 컨테이너 오케스트레이션 설정 파일",
            "로드 밸런서 트래픽 분산 처리 로직 및 웹 방화벽(WAF) 차단 규칙",
            "회사 핵심 서비스의 백엔드 비즈니스 로직 및 미공개 소스코드",
            "신규 개발 중인 사용자 맞춤형 추천 알고리즘의 가중치 계산 로직",
            "결제 모듈 연동 및 금융 데이터 암호화 처리 코드 설계",
            "상용 배포 전 단계의 테스트 서버 소스코드 및 환경 변수(env) 파일",
            "대형 언어 모델(LLM) 통합 및 RAG(검색 증강 생성) 파이프라인 내부 구현 코드",
            "데이터베이스 최고 관리자 계정 접근 권한 및 인증 패스워드",
            "데이터베이스 테이블 스키마 설계도 및 ERD(Entity-Relationship Diagram) 구조",
            "외부 API 연동을 위한 비공개 발급 키 및 시스템 시크릿 토큰",
            "관리자 전용 백오피스 시스템 접근 API 엔드포인트 및 파라미터 규격",
            "시스템 사용자 인증용 JWT 토큰 생성 및 세션 유지 처리 로직",
            "사내 시스템 대상 모의해킹 결과 및 보안 취약점 분석 리포트",
            "버퍼 오버플로우(Buffer Overflow) 및 포맷 스트링 등 시스템 메모리 취약점 정보",
            "소스코드 정적 분석 도구에서 발견된 크리티컬 에러 및 보안 결함 내역",
            "보안 솔루션 우회 기법 및 내부망 침투(Penetration) 테스트 시나리오",
            "침해 사고 대응 이력 및 서버 포렌식 침입 탐지 로그 데이터",
            "서비스 가입 고객의 고유 식별 정보 및 개인정보 데이터베이스",
            "스마트 의료 가이드 및 헬스케어 서비스의 민감한 사용자 건강/진단 정보",
            "무선 통신 기기 및 IoT 센서의 RF 핑거프린팅(RF Fingerprinting) 추출 데이터",
            "임직원 인사 평가, 사내 조직도 및 보안 등급별 권한 부여 내역",
            "VIP 고객 명단 및 타겟 마케팅 상세 분석 원본 데이터",
            "타사 인수합병(M&A) 검토 및 미공개 투자 전략 내부 문서",
            "협력사와의 기밀 유지 협약(NDA) 및 주요 API 제공 계약 조건",
            "차기 분기 재무 제표, 서버 유지보수 예산 및 인건비 상세 내역",
            "신규 런칭 예정 서비스의 비공개 기획안 및 UI/UX 프로토타입",
            "침해 사고 발생 시 서버 다운타임 대처 및 BCP(업무 연속성 계획) 매뉴얼"
        ]
        
        # 모델을 이용해 30개의 기준 문장을 벡터(Tensor)로 변환하여 메모리에 상주
        self.ref_embeddings = self.model.encode(self.sensitive_references, convert_to_tensor=True)
        self.threshold = 0.70 

        # 3. 정형 데이터 11종 (클라우드 키, JWT, DB 등) 패턴
        self.regex_patterns = {
            "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
            "PHONE": re.compile(r"01[016789]-?\d{3,4}-?\d{4}"),
            "NAME": re.compile(r"(?:이름|성명|name)\s*[:=]\s*([가-힣]{2,4})"),
            "AWS_KEY": re.compile(r"(?<![A-Z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])"),
            "GCP_KEY": re.compile(r"(?<![A-Za-z0-9_-])AIza[0-9A-Za-z\-_]{35}(?![A-Za-z0-9_-])"),
            "AZURE_KEY": re.compile(r"(?i)(?:azure|storage)[^\w]*([a-zA-Z0-9+/]{86}==)"),
            "JWT_TOKEN": re.compile(r"ey[a-zA-Z0-9_-]+\.ey[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+"),
            "DB_URL": re.compile(r"(?:postgres|mysql|mongodb|redis|sqlite|oracle)://[a-zA-Z0-9_-]+:[^@\s]+@[a-zA-Z0-9.-]+:\d+/[a-zA-Z0-9_]+"),
            "PRIVATE_KEY": re.compile(r"-----BEGIN (?:RSA|OPENSSH|DSA|EC|PGP)?\s*PRIVATE KEY-----[\s\S]+?-----END (?:RSA|OPENSSH|DSA|EC|PGP)?\s*PRIVATE KEY-----"),
            "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
            "ZIP_CODE": re.compile(r"\b(?:[0-9]{5}(?:-[0-9]{4})?|[A-Z][0-9][A-Z]\s?[0-9][A-Z][0-9]|[A-Z]{1,2}[0-9R][0-9A-Z]?\s?[0-9][A-Z]{2})\b", re.IGNORECASE)
        }

        # 4. 양방향 복원을 위한 메모리 금고
        self.vault = {}
        logging.info("===== CodeSentinAI: 양방향 마스킹 게이트웨이 가동 완료 =====")

    def request(self, flow: http.HTTPFlow) -> None:
        """ [나가는 트래픽] 사용자의 프롬프트를 가로채서 마스킹 """
        try:
            if flow.request.method == "POST" and flow.request.content:
                if "chatgpt.com" in flow.request.pretty_host or "api.openai.com" in flow.request.pretty_host:
                    
                    # 1. JSON 포맷 헤더 검사
                    content_type = flow.request.headers.get("Content-Type", "")
                    if "application/json" not in content_type:
                        return
                    
                    body = flow.request.text 
                    
                    # 2. 빈 텍스트 검사
                    if not body or not body.strip(): 
                        return
                        
                    # 3. JSON 파싱 에러(char 0 등) 무시
                    try:
                        data = json.loads(body)
                    except json.JSONDecodeError:
                        return

                    is_modified = False
                    
                    if "messages" in data:
                        for message in data["messages"]:
                            if "content" in message and "parts" in message["content"]:
                                for i, part in enumerate(message["content"]["parts"]):
                                    if isinstance(part, str):
                                        masked_text = self.mask_and_store(part)
                                        if part != masked_text:
                                            message["content"]["parts"][i] = masked_text
                                            is_modified = True
                                            
                    if is_modified:
                        logging.info("[요청 변조] 기밀 데이터를 안전한 토큰으로 치환하여 서버로 전송합니다.")
                        flow.request.text = json.dumps(data)
                        
        except Exception as e:
            logging.error(f"[요청 처리 에러] {e}")

    def response(self, flow: http.HTTPFlow) -> None:
        """ [들어오는 트래픽] ChatGPT의 답변을 가로채서 원본 데이터로 안전하게 복원 """
        try:
            if "chatgpt.com" in flow.request.pretty_host or "api.openai.com" in flow.request.pretty_host:
                if flow.response and flow.response.text:
                    original_body = flow.response.text
                    restored_body = original_body
                    
                    for placeholder, original_text in self.vault.items():
                        if placeholder in restored_body:
                            safe_text = json.dumps(original_text, ensure_ascii=False)[1:-1]
                            restored_body = restored_body.replace(placeholder, safe_text)
                            logging.warning(f"[응답 복원 성공] {placeholder} -> 원본 복원됨!")
                    
                    if original_body != restored_body:
                        flow.response.text = restored_body
        except Exception as e:
            logging.error(f"[응답 복원 에러] {e}")

    def mask_and_store(self, text: str) -> str:
        """ 1단계: 정규식 마스킹 -> 2단계: SBERT 마스킹 """
        masked_text = text
        
        # 1단계: 확장된 정규식 마스킹 (안전한 치환 방식 적용)
        for label, pattern in self.regex_patterns.items():
            matches = list(pattern.finditer(masked_text))
            # 인덱스 꼬임 방지를 위해 뒤에서부터 치환
            for match in reversed(matches):
                original_value = match.group(1) if match.groups() else match.group(0)
                start_idx = match.start(1) if match.groups() else match.start()
                end_idx = match.end(1) if match.groups() else match.end()
                
                placeholder = f"[[{label}_{str(uuid.uuid4())[:4]}]]"
                self.vault[placeholder] = original_value
                
                masked_text = masked_text[:start_idx] + placeholder + masked_text[end_idx:]
                logging.info(f"[Regex 탐지] {label} 토큰 발행 완료")
                
        # 2단계: SBERT 기반 문맥 마스킹
        sentences = masked_text.replace('\n', '. ').split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            # 정규식으로 이미 토큰화된 문장이나 짧은 문장은 SBERT 검사 스킵
            if len(sentence) < 2 or "[[" in sentence: 
                continue

            input_embedding = self.model.encode(sentence, convert_to_tensor=True)
            cosine_scores = util.cos_sim(input_embedding, self.ref_embeddings)[0]
            max_score = float(cosine_scores.max())
            
            if max_score >= self.threshold:
                placeholder = f"[[BUSINESS_LOGIC_{str(uuid.uuid4())[:4]}]]"
                self.vault[placeholder] = sentence
                masked_text = masked_text.replace(sentence, placeholder)
                logging.info(f"[SBERT 탐지] 유사도 {max_score:.2f} -> {placeholder}")

        return masked_text

addons = [
    CodeSentinAI()
]