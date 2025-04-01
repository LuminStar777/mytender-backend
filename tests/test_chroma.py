import pytest
from services.embedding import text_to_chromadb
from config import embedder


@pytest.mark.asyncio
async def test_chunk_and_chroma_embed():
    """Test embedding text to chromadb"""

    long_text = """# Comprehensive Safety and Operations Manual

## 1. General Safety Guidelines
- Always wear appropriate Personal Protective Equipment (PPE)
- Follow all posted safety signs and instructions
- Report any hazards or incidents immediately
- Maintain clean and organized work areas
- Attend all required safety training sessions
- Keep emergency exits clear at all times

## 2. Emergency Response Procedures
### 2.1 Fire Emergency
1. Activate nearest fire alarm
2. Call emergency services (Extension 555)
3. Follow evacuation procedures
4. Meet at designated assembly point
5. Perform head count of all personnel
6. Wait for all-clear signal from emergency responders

### 2.2 Medical Emergency
- Call first aid responder immediately
- Secure the area to prevent further incidents
- Provide assistance if qualified
- Document incident details thoroughly
- Follow up with incident investigation
- Update procedures if necessary

## 3. Equipment Safety Protocols
* Inspect all equipment before each use
* Only operate equipment you're trained and certified to use
* Report malfunctions to maintenance department immediately
* Follow lockout/tagout procedures strictly
* Maintain equipment logs accurately
* Schedule regular maintenance checks

## 4. Chemical Safety Guidelines
- Store chemicals according to compatibility chart
- Use appropriate containment methods
- Know location of Safety Data Sheets (SDS)
- Use proper handling techniques
- Wear specific PPE for each chemical
- Report spills immediately

## 5. Environmental Management
* Implement proper waste disposal procedures
* Follow energy conservation guidelines
* Monitor water usage and management
* Control emissions according to regulations
* Document all environmental incidents
* Conduct regular environmental audits

## 6. Risk Assessment Matrix

| Likelihood/Severity | Negligible | Minor | Moderate | Major | Catastrophic |
|-------------------|------------|-------|-----------|-------|--------------|
| Almost Certain    | Medium     | High  | High      | Extreme| Extreme     |
| Likely            | Medium     | Medium| High      | High   | Extreme     |
| Possible          | Low        | Medium| Medium    | High   | Extreme     |
| Unlikely          | Low        | Low   | Medium    | High   | High        |
| Rare              | Low        | Low   | Medium    | Medium | High        |

## 7. PPE Requirements by Area

| Area              | Hard Hat | Safety Glasses | Steel Toe Boots | Hearing Protection | Respirator |
|-------------------|----------|----------------|-----------------|-------------------|------------|
| Production Floor  | Required | Required       | Required        | Required          | As needed  |
| Laboratory        | Optional | Required       | Required        | Optional          | As needed  |
| Warehouse        | Required | Required       | Required        | Optional          | No         |
| Office Areas     | No       | No             | No              | No                | No         |
| Loading Dock     | Required | Required       | Required        | Optional          | No         |

## 8. Emergency Contact Information
### 8.1 Internal Contacts
- Emergency Response Team: Ext. 555
- Security Office: Ext. 444
- Facility Manager: Ext. 333
- Health & Safety Officer: Ext. 222
- Environmental Coordinator: Ext. 111

### 8.2 External Emergency Services
- Fire Department: 911
- Police: 911
- Ambulance: 911
- Poison Control: 1-800-222-1222
- Environmental Agency: 1-800-424-8802

## 9. Training Requirements
* Initial Safety Orientation
* Annual Safety Refresher
* Equipment-Specific Training
* Emergency Response Training
* First Aid and CPR
* Hazardous Materials Handling
* Environmental Compliance

## 10. Documentation and Reporting
- Maintain accurate incident reports
- Update safety procedures regularly
- Record all training completions
- Document equipment inspections
- Keep environmental compliance records
- File monthly safety statistics

This comprehensive document serves as a general guide and should be used in conjunction with specific departmental procedures and local regulations. All personnel must familiarize themselves with these guidelines and acknowledge their understanding through signed documentation."""

    await text_to_chromadb(
        text=long_text,
        user="tests/chroma_db",
        collection="QHSE_DocumentationFORWARDSLASHStandard_Operating_Prok",
        user_name="test_user",
        mode="qa",
        embedding=embedder,
        metadata={
            "filename": "safety_operations_manual.txt",
            "upload_date": "14/01/2025",
            "uploaded_by": "test_user"
        },
    )
