"""
Test module for the tender bid generation functions. i.e opportunity info, compliance requirements, 
exec summary, cover letter
"""

import pytest
from services.chain import (
    get_compliance_requirements,
    get_exec_summary,
    get_opportunity_information,
    get_cover_letter,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def sample_documents():
    """
    Fixture that provides sample documents for tests, containing realistic extracts from tender documents.

    Returns:
        list: A list of dictionaries containing sample document texts from various sections of tender documents.
    """
    return [
        {
            "text": """
Request for Proposal (RFP) - Cloud-Based Enterprise Resource Planning (ERP) System

1. Introduction
   Acme Corporation is seeking proposals for the implementation of a comprehensive cloud-based Enterprise Resource Planning (ERP) system. This system will integrate our core business processes, including finance, human resources, supply chain management, and customer relationship management.

2. Project Scope
   - Implementation of a cloud-based ERP system
   - Data migration from existing systems
   - Integration with current third-party applications
   - User training and documentation
   - Ongoing support and maintenance

3. Key Requirements
   - Financial Management: General ledger, accounts payable/receivable, asset management
   - Human Resources: Payroll, time and attendance, talent management
   - Supply Chain: Inventory management, procurement, order fulfillment
   - Customer Relationship Management: Sales automation, customer service, marketing campaigns
   - Reporting and Analytics: Customizable dashboards, real-time reporting capabilities

4. Timeline
   - RFP Release Date: August 1, 2024
   - Proposal Due Date: September 15, 2024
   - Vendor Selection: October 31, 2024
   - Project Kickoff: January 2, 2025
   - Go-Live Target: July 1, 2025

5. Budget
   The anticipated budget range for this project is $2-3 million, including software licensing, implementation services, and first-year support.
"""
        },
        {
            "text": """
Compliance and Technical Requirements

1. System Architecture
   - Cloud-based SaaS solution
   - Multi-tenant architecture with dedicated database for Acme Corporation
   - 99.9% uptime guarantee

2. Security and Compliance
   - SOC 2 Type II certified
   - GDPR and CCPA compliant
   - Multi-factor authentication
   - Role-based access control
   - Encryption of data at rest and in transit

3. Integration Capabilities
   - RESTful API for third-party integrations
   - Support for single sign-on (SSO) using SAML 2.0
   - Ability to integrate with existing data warehouse for BI reporting

4. Scalability and Performance
   - Ability to handle 1000+ concurrent users
   - Scalable to accommodate 25% year-over-year growth
   - Response time < 2 seconds for 95% of transactions

5. Data Migration
   - Vendor must provide a detailed data migration strategy
   - Support for automated and manual data cleansing tools

6. Training and Support
   - Comprehensive training program for end-users and IT staff
   - 24/7 technical support with maximum 2-hour response time for critical issues

7. Compliance with Industry Standards
   - GAAP compliance for financial modules
   - ISO 27001 certification for information security management

Vendors must explicitly address how their solution meets each of these requirements in their proposal.
"""
        },
    ]


@pytest.mark.asyncio
async def test_get_compliance_requirements_integrated(sample_documents):
    """
    Test the get_compliance_requirements function with minimal mocking.
    This test uses the actual chain invocation to test the function's behavior.
    """
    # Call the function with sample documents
    bid_id = '66c08bd6c45a0408088d0211'
    _ = await get_compliance_requirements(bid_id, "adminuser")


@pytest.mark.asyncio
async def test_get_opportunity_information_integrated(sample_documents):
    """
    Test the get_opportunity_information function with minimal mocking.
    This test uses the actual chain invocation to test the function's behavior.
    """
    bid_id = '66c08bd6c45a0408088d0211'
    _ = await get_opportunity_information(bid_id)


@pytest.mark.asyncio
async def test_get_exec_summary_integrated(sample_documents):
    """
    Test the get_exec_summary function with minimal mocking.
    This test uses the actual chain invocation to test the function's behavior.
    """
    result = await get_exec_summary(sample_documents)
    # Assertions
    assert isinstance(result, str)
    assert len(result) > 0
    assert "\n\n" in result  # Check for double newlines between paragraphs
    assert "Error: Unable to generate executive summary." not in result


@pytest.mark.asyncio
async def test_get_cover_letter_integrated(sample_documents):
    """
    Test the get_cover_letter function with minimal mocking.
    This test uses the actual chain invocation to test the function's behavior.
    """
    # Call the function with sample documents
    result = await get_cover_letter(sample_documents)
    # Assertions
    assert isinstance(result, str)
    assert len(result) > 0
    assert "\n\n" in result  # Check for double newlines between paragraphs
    # Verify that load_user_config was called correctly
    assert "Error: Unable to generate cover letter." not in result
