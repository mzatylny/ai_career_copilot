from app.ai_services import _ground_chat_response
from app.models import ChatResponse, SourceChunk


def _context():
    return [
        {
            "chunk_id": "trusted-1",
            "source": "resume.pdf",
            "page": 2,
            "text": "Built a production FastAPI service.",
            "relevance_score": 0.88,
        }
    ]


def test_grounding_drops_hallucinated_source_ids():
    response = ChatResponse(
        answer="Grounded answer",
        sources=[
            SourceChunk(
                source="invented.pdf",
                page=99,
                chunk_id="hallucinated",
                snippet="Invented evidence",
                relevance_score=1.0,
            )
        ],
        confidence="high",
        follow_up_questions=[],
    )

    grounded = _ground_chat_response(response, _context())

    assert [source.chunk_id for source in grounded.sources] == ["trusted-1"]
    assert grounded.sources[0].source == "resume.pdf"
    assert grounded.sources[0].page == 2


def test_grounding_replaces_model_metadata_with_retrieval_metadata():
    response = ChatResponse(
        answer="Grounded answer",
        sources=[
            SourceChunk(
                source="wrong.pdf",
                page=50,
                chunk_id="trusted-1",
                snippet="Wrong snippet",
                relevance_score=0.1,
            )
        ],
        confidence="high",
        follow_up_questions=[],
    )

    grounded = _ground_chat_response(response, _context())

    assert grounded.sources[0].source == "resume.pdf"
    assert grounded.sources[0].snippet == "Built a production FastAPI service."
    assert grounded.sources[0].relevance_score == 0.88


def test_grounding_downgrades_confidence_without_real_context():
    response = ChatResponse(
        answer="No evidence",
        sources=[],
        confidence="high",
        follow_up_questions=[],
    )

    grounded = _ground_chat_response(response, [])

    assert grounded.sources == []
    assert grounded.confidence == "low"
