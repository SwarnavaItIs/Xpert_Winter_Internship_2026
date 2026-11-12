import json
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile
from tools import day1


class Day1Tests(unittest.TestCase):
    def test_language_profile_is_script_based(self):
        self.assertEqual(day1.script_bucket('Hello नमस्ते'), 'Latin+Devanagari')
        self.assertEqual(day1.script_bucket('বাংলা'), 'Bengali')
        self.assertEqual(day1.script_bucket('123'), 'no_letters')

    def test_archive_rejects_traversal_before_extraction(self):
        root = Path('.private/tests/day1-archive').resolve()
        root.mkdir(parents=True, exist_ok=True)
        archive = root/'bad.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('../escaped.txt', 'synthetic')
        with self.assertRaises(ValueError):
            day1.extract_verified(archive, root/'target')
        self.assertFalse((root/'escaped.txt').exists())

    def test_split_is_repeatable_and_refuses_input_changes_without_reading_answers(self):
        import uuid
        root = Path('.private/tests/day1-split').resolve()/uuid.uuid4().hex
        root.mkdir(parents=True)
        chats = [{'chat_id': 'test-'+str(i), 'participants': ['ME', 'synthetic-person']} for i in range(10)]
        messages = [{'id': 'msg-'+str(i), 'chat_id': 'test-'+str(i)} for i in range(10)]
        for name, records in [('chats', chats), ('messages', messages)]:
            (root/(name+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
        (root/'seed.json').write_text('{}', encoding='utf-8')
        original = Path.open
        def guarded(path, *args, **kwargs):
            if 'answers' in path.parts:
                raise AssertionError('Split accessed labels')
            return original(path, *args, **kwargs)
        with patch.object(day1, 'PRIVATE', root/'private'), patch.object(Path, 'open', guarded):
            first = day1.freeze_split(root)
            second = day1.freeze_split(root)
            self.assertEqual(first, second)
            self.assertEqual(first['validation_chats'], 2)
            with (root/'messages.jsonl').open('a', encoding='utf-8') as f:
                f.write('\n')
            with self.assertRaises(ValueError):
                day1.freeze_split(root)


if __name__ == '__main__':
    unittest.main()
