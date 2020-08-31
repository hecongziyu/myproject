from kg.persist import KnowledgeEntity

def test_get_all_entity(kge):
    result = get_all_entity()
    print(result)


if __name__ == '__main__':
    kge = KnowledgeEntity()

    test_get_all_entity(kge)